from __future__ import annotations

import logging
import re
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger("chatpdf")

BASE_URL = "https://mineru.net/api/v4"


def process_pdf_via_mineru(
    pdf_url: str,
    *,
    output_root: Path,
    api_key: str,
    poll_interval: int = 5,
    timeout_seconds: int = 600,
) -> Path:
    """
    Submit a PDF to MinerU, poll until complete, and return the resulting markdown path.
    """
    headers = _mineru_headers(api_key)
    parsed_url = urlparse(pdf_url)
    original_name = Path(unquote(parsed_url.path)).name or "document.pdf"
    stem = _sanitize_basename(Path(original_name).stem)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    task_label = f"{stem}_{timestamp}"
    target_dir = output_root / task_label
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "url": pdf_url,
        "is_ocr": False,
        "enable_formula": True,
        "enable_table": True,
    }
    logger.info("Submitting MinerU extraction task for %s", pdf_url)

    submission = _request_with_retries(
        "POST",
        f"{BASE_URL}/extract/task",
        json=payload,
        headers=headers,
    ).json()
    if submission.get("code") != 0:
        raise RuntimeError(f"MinerU task submission failed: {submission}")

    task_id = submission["data"]["task_id"]
    logger.info("MinerU task created: %s", task_id)

    deadline = time.time() + timeout_seconds
    task_info: Dict[str, str] | None = None
    while time.time() < deadline:
        task_data = _request_with_retries(
            "GET",
            f"{BASE_URL}/extract/task/{task_id}",
            headers=headers,
        ).json()
        if task_data.get("code") != 0:
            raise RuntimeError(f"MinerU task query failed: {task_data}")

        task_info = task_data["data"]
        state = task_info.get("state")
        logger.info("MinerU task %s state: %s", task_id, state)

        if state == "done":
            break
        if state == "failed":
            raise RuntimeError(
                f"MinerU task {task_id} failed: {task_info.get('err_msg', 'unknown reason')}"
            )
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Timed out waiting for MinerU task {task_id} to finish")

    zip_url = task_info.get("full_zip_url") if task_info else None
    if not zip_url:
        raise RuntimeError("MinerU task completed but no result package URL provided")

    with TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "result.zip"
        _download_file(zip_url, zip_path)
        markdown_path = _extract_markdown_from_zip(zip_path, target_dir)

    pdf_destination = target_dir / f"{task_label}.pdf"
    try:
        _download_file(pdf_url, pdf_destination)
    except Exception as exc:
        logger.warning("Failed to download original PDF %s: %s", pdf_url, exc)

    logger.info(
        "MinerU processing complete. Markdown: %s, PDF: %s",
        markdown_path,
        pdf_destination,
    )
    return markdown_path


def _mineru_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request_with_retries(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: int = 2,
    **kwargs,
) -> requests.Response:
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            if response.status_code == 200:
                return response
            logger.warning(
                "MinerU API %s %s returned status %s (attempt %s/%s)",
                method,
                url,
                response.status_code,
                attempt,
                max_retries,
            )
        except requests.RequestException as exc:
            logger.warning(
                "MinerU API %s %s request error on attempt %s/%s: %s",
                method,
                url,
                attempt,
                max_retries,
                exc,
            )
        if attempt < max_retries:
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"MinerU API request failed after {max_retries} attempts: {url}")


def _sanitize_basename(name: str) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    return stem or "document"


def _download_file(url: str, destination: Path) -> None:
    logger.info("Downloading file from %s to %s", url, destination)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)


def _extract_markdown_from_zip(zip_path: Path, target_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    markdown_files = sorted(target_dir.rglob("*.md"))
    if not markdown_files:
        raise FileNotFoundError("No markdown file found in MinerU result package")
    markdown_files.sort(key=lambda item: item.stat().st_size, reverse=True)
    selected = markdown_files[0]
    logger.info("Selected markdown file %s from MinerU results", selected)
    return selected


def process_local_files_via_mineru(
    file_paths: list[Path],
    *,
    output_root: Path,
    api_key: str,
    poll_interval: int = 5,
    timeout_seconds: int = 600,
    model_version: str = "vlm",
) -> list[Path]:
    """
    Submit local files to MinerU for batch processing, poll until complete, and return the resulting markdown paths.
    """
    headers = _mineru_headers(api_key)
    
    # Prepare file data for batch upload URL request
    files_data = []
    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        files_data.append({
            "name": file_path.name,
            "data_id": file_path.stem
        })
    
    # Request batch upload URLs
    payload = {
        "files": files_data,
        "model_version": model_version
    }
    
    logger.info("Requesting batch upload URLs for %d files", len(file_paths))
    
    response = _request_with_retries(
        "POST",
        f"{BASE_URL}/file-urls/batch",
        json=payload,
        headers=headers,
    ).json()
    
    if response.get("code") != 0:
        raise RuntimeError(f"MinerU batch upload URL request failed: {response}")
    
    batch_id = response["data"]["batch_id"]
    file_urls = response["data"]["file_urls"]
    
    logger.info("Batch ID: %s, received %d upload URLs", batch_id, len(file_urls))
    
    # Upload files to the provided URLs
    for i, (file_path, upload_url) in enumerate(zip(file_paths, file_urls)):
        logger.info("Uploading file %d/%d: %s", i + 1, len(file_paths), file_path)
        try:
            with open(file_path, 'rb') as f:
                upload_response = requests.put(upload_url, data=f, timeout=120)
                if upload_response.status_code != 200:
                    logger.error("File upload failed for %s: %s", file_path, upload_response.status_code)
                    raise RuntimeError(f"Failed to upload file {file_path}: {upload_response.status_code}")
                logger.info("Successfully uploaded %s", file_path)
        except Exception as e:
            logger.error("Error uploading file %s: %s", file_path, e)
            raise RuntimeError(f"Error uploading file {file_path}: {e}")
    
    # Wait for processing to complete and get results
    return _wait_for_batch_completion(
        batch_id=batch_id,
        file_paths=file_paths,
        output_root=output_root,
        api_key=api_key,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
    )


def get_batch_results(
    batch_id: str,
    *,
    api_key: str,
) -> dict:
    """
    Get batch processing results by batch_id.
    """
    headers = _mineru_headers(api_key)
    
    logger.info("Getting batch results for batch_id: %s", batch_id)
    
    response = _request_with_retries(
        "GET",
        f"{BASE_URL}/extract-results/batch/{batch_id}",
        headers=headers,
    ).json()
    
    if response.get("code") != 0:
        raise RuntimeError(f"MinerU batch results query failed: {response}")
    
    return response["data"]


def _wait_for_batch_completion(
    batch_id: str,
    file_paths: list[Path],
    output_root: Path,
    api_key: str,
    poll_interval: int,
    timeout_seconds: int,
) -> list[Path]:
    """
    Wait for batch processing to complete and download results.
    """
    deadline = time.time() + timeout_seconds
    markdown_paths = []
    
    while time.time() < deadline:
        batch_data = get_batch_results(batch_id, api_key=api_key)
        
        logger.info("Batch %s status: %s", batch_id, batch_data.get("status"))
        logger.debug("Batch data: %s", batch_data)
        
        # Check if all tasks are done - API uses "extract_result" instead of "tasks"
        tasks = batch_data.get("extract_result", [])
        if not tasks:
            logger.warning("No tasks found in batch response, checking for direct status")
            # Check if batch is directly completed
            status = batch_data.get("status")
            if status == "completed":
                # Try to process based on available data
                markdown_paths = _process_completed_batch(batch_data, file_paths, output_root, api_key)
                if markdown_paths:
                    return markdown_paths
            logger.warning("No tasks found in batch response")
            time.sleep(poll_interval)
            continue
        
        all_done = True
        for task in tasks:
            state = task.get("state")
            if state not in ["done", "failed"]:
                all_done = False
                break
        
        if all_done:
            # Process completed tasks
            for task in tasks:
                if task.get("state") == "done":
                    file_name = task.get("file_name")
                    original_file = next((fp for fp in file_paths if fp.name == file_name), None)
                    if original_file and task.get("full_zip_url"):
                        markdown_path = _process_single_task_result(
                            task, original_file, output_root, api_key
                        )
                        markdown_paths.append(markdown_path)
            break
        
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Timed out waiting for batch {batch_id} to finish")
    
    return markdown_paths


def _process_completed_batch(
    batch_data: dict,
    file_paths: list[Path],
    output_root: Path,
    api_key: str,
) -> list[Path]:
    """
    Process a completed batch when no task information is available.
    """
    markdown_paths = []
    
    # Try to get result URLs from batch data
    result_urls = batch_data.get("result_urls", [])
    if not result_urls:
        logger.warning("No result URLs found in batch data")
        return []
    
    for i, file_path in enumerate(file_paths):
        if i < len(result_urls):
            result_url = result_urls[i]
            try:
                stem = _sanitize_basename(file_path.stem)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                task_label = f"{stem}_{timestamp}"
                target_dir = output_root / task_label
                target_dir.mkdir(parents=True, exist_ok=True)
                
                with TemporaryDirectory() as tmpdir:
                    zip_path = Path(tmpdir) / "result.zip"
                    _download_file(result_url, zip_path)
                    markdown_path = _extract_markdown_from_zip(zip_path, target_dir)
                
                # Copy original file to output directory
                original_destination = target_dir / f"{task_label}{file_path.suffix}"
                try:
                    shutil.copy2(file_path, original_destination)
                except Exception as exc:
                    logger.warning("Failed to copy original file %s: %s", file_path, exc)
                
                logger.info(
                    "Batch processing complete for %s. Markdown: %s",
                    file_path.name,
                    markdown_path,
                )
                markdown_paths.append(markdown_path)
                
            except Exception as e:
                logger.error("Failed to process result for %s: %s", file_path.name, e)
    
    return markdown_paths


def _process_single_task_result(
    task: dict,
    original_file: Path,
    output_root: Path,
    api_key: str,
) -> Path:
    """
    Process a single completed task result.
    """
    stem = _sanitize_basename(original_file.stem)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    task_label = f"{stem}_{timestamp}"
    target_dir = output_root / task_label
    target_dir.mkdir(parents=True, exist_ok=True)
    
    zip_url = task.get("full_zip_url")
    if not zip_url:
        raise RuntimeError(f"No result package URL for task {task.get('task_id')}")
    
    with TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "result.zip"
        _download_file(zip_url, zip_path)
        markdown_path = _extract_markdown_from_zip(zip_path, target_dir)
    
    # Copy original file to output directory
    original_destination = target_dir / f"{task_label}{original_file.suffix}"
    try:
        shutil.copy2(original_file, original_destination)
    except Exception as exc:
        logger.warning("Failed to copy original file %s: %s", original_file, exc)
    
    logger.info(
        "Batch processing complete for %s. Markdown: %s",
        original_file.name,
        markdown_path,
    )
    return markdown_path


def process_urls_via_mineru(
    urls: list[str],
    *,
    output_root: Path,
    api_key: str,
    poll_interval: int = 5,
    timeout_seconds: int = 600,
    model_version: str = "vlm",
) -> list[Path]:
    """
    Submit URLs to MinerU for batch processing, poll until complete, and return the resulting markdown paths.
    """
    headers = _mineru_headers(api_key)
    
    # Prepare URL data for batch task submission
    files_data = []
    for url in urls:
        # Extract filename from URL for data_id
        parsed_url = urlparse(url)
        filename = Path(unquote(parsed_url.path)).name or "document.pdf"
        data_id = _sanitize_basename(Path(filename).stem)
        files_data.append({
            "url": url,
            "data_id": data_id
        })
    
    # Submit batch task
    payload = {
        "files": files_data,
        "model_version": model_version
    }
    
    logger.info("Submitting batch URL processing for %d URLs", len(urls))
    
    response = _request_with_retries(
        "POST",
        f"{BASE_URL}/extract/task/batch",
        json=payload,
        headers=headers,
    ).json()
    
    if response.get("code") != 0:
        raise RuntimeError(f"MinerU batch URL submission failed: {response}")
    
    batch_id = response["data"]["batch_id"]
    logger.info("Batch ID: %s", batch_id)
    
    # Wait for processing to complete and get results
    return _wait_for_url_batch_completion(
        batch_id=batch_id,
        urls=urls,
        output_root=output_root,
        api_key=api_key,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
    )


def _wait_for_url_batch_completion(
    batch_id: str,
    urls: list[str],
    output_root: Path,
    api_key: str,
    poll_interval: int,
    timeout_seconds: int,
) -> list[Path]:
    """
    Wait for URL batch processing to complete and download results.
    """
    deadline = time.time() + timeout_seconds
    markdown_paths = []
    
    while time.time() < deadline:
        batch_data = get_batch_results(batch_id, api_key=api_key)
        
        logger.info("Batch %s status: %s", batch_id, batch_data.get("status"))
        logger.debug("Batch data: %s", batch_data)
        
        # Check if all tasks are done - API uses "extract_result" instead of "tasks"
        tasks = batch_data.get("extract_result", [])
        if not tasks:
            logger.warning("No tasks found in batch response, checking for direct status")
            # Check if batch is directly completed
            status = batch_data.get("status")
            if status == "completed":
                # Try to process based on available data
                markdown_paths = _process_completed_url_batch(batch_data, urls, output_root, api_key)
                if markdown_paths:
                    return markdown_paths
            logger.warning("No tasks found in batch response")
            time.sleep(poll_interval)
            continue
        
        all_done = True
        for task in tasks:
            state = task.get("state")
            if state not in ["done", "failed"]:
                all_done = False
                break
        
        if all_done:
            # Process completed tasks
            for task in tasks:
                if task.get("state") == "done":
                    file_name = task.get("file_name")
                    original_url = next((url for url in urls if file_name in url), None)
                    if original_url and task.get("full_zip_url"):
                        markdown_path = _process_single_url_task_result(
                            task, original_url, output_root, api_key
                        )
                        markdown_paths.append(markdown_path)
            break
        
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Timed out waiting for batch {batch_id} to finish")
    
    return markdown_paths


def _process_completed_url_batch(
    batch_data: dict,
    urls: list[str],
    output_root: Path,
    api_key: str,
) -> list[Path]:
    """
    Process a completed URL batch when no task information is available.
    """
    markdown_paths = []
    
    # Try to get result URLs from batch data
    result_urls = batch_data.get("result_urls", [])
    if not result_urls:
        logger.warning("No result URLs found in batch data")
        return []
    
    for i, url in enumerate(urls):
        if i < len(result_urls):
            result_url = result_urls[i]
            try:
                parsed_url = urlparse(url)
                filename = Path(unquote(parsed_url.path)).name or "document.pdf"
                stem = _sanitize_basename(Path(filename).stem)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                task_label = f"{stem}_{timestamp}"
                target_dir = output_root / task_label
                target_dir.mkdir(parents=True, exist_ok=True)
                
                with TemporaryDirectory() as tmpdir:
                    zip_path = Path(tmpdir) / "result.zip"
                    _download_file(result_url, zip_path)
                    markdown_path = _extract_markdown_from_zip(zip_path, target_dir)
                
                # Download original file to output directory
                original_destination = target_dir / f"{task_label}.pdf"
                try:
                    _download_file(url, original_destination)
                except Exception as exc:
                    logger.warning("Failed to download original file from %s: %s", url, exc)
                
                logger.info(
                    "URL batch processing complete for %s. Markdown: %s",
                    filename,
                    markdown_path,
                )
                markdown_paths.append(markdown_path)
                
            except Exception as e:
                logger.error("Failed to process result for %s: %s", url, e)
    
    return markdown_paths


def _process_single_url_task_result(
    task: dict,
    original_url: str,
    output_root: Path,
    api_key: str,
) -> Path:
    """
    Process a single completed URL task result.
    """
    parsed_url = urlparse(original_url)
    filename = Path(unquote(parsed_url.path)).name or "document.pdf"
    stem = _sanitize_basename(Path(filename).stem)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    task_label = f"{stem}_{timestamp}"
    target_dir = output_root / task_label
    target_dir.mkdir(parents=True, exist_ok=True)
    
    zip_url = task.get("full_zip_url")
    if not zip_url:
        raise RuntimeError(f"No result package URL for task {task.get('task_id')}")
    
    with TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "result.zip"
        _download_file(zip_url, zip_path)
        markdown_path = _extract_markdown_from_zip(zip_path, target_dir)
    
    # Download original file to output directory
    original_destination = target_dir / f"{task_label}.pdf"
    try:
        _download_file(original_url, original_destination)
    except Exception as exc:
        logger.warning("Failed to download original file from %s: %s", original_url, exc)
    
    logger.info(
        "URL batch processing complete for %s. Markdown: %s",
        filename,
        markdown_path,
    )
    return markdown_path


__all__ = [
    "process_pdf_via_mineru",
    "process_local_files_via_mineru",
    "process_urls_via_mineru",
    "get_batch_results"
]
