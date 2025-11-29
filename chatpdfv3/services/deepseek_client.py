from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger("chatpdf")

# DeepSeek 价格 (元/百万tokens)
DEEPSEEK_PRICE_INPUT_CACHE_HIT = 0.2  # 缓存命中
DEEPSEEK_PRICE_INPUT_CACHE_MISS = 2.0  # 缓存未命中
DEEPSEEK_PRICE_OUTPUT = 3.0  # 输出

# 假设缓存命中率为 0%，使用缓存未命中价格
PRICE_INPUT_PER_1K = DEEPSEEK_PRICE_INPUT_CACHE_MISS / 1000
PRICE_OUTPUT_PER_1K = DEEPSEEK_PRICE_OUTPUT / 1000


def create_deepseek_client() -> OpenAI:
    """
    创建 DeepSeek API 客户端
    """
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


def post_with_retries_deepseek(
    client: OpenAI,
    model: str,
    messages: list[Dict[str, str]],
    *,
    temperature: float = 1.0,
    max_retries: int = 4,
    base_delay: int = 1,
) -> Optional[Any]:
    """
    DeepSeek API 调用包装器，包含重试机制和错误处理
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False
            )
            
            _log_usage_deepseek(response)
            return response
            
        except Exception as exc:
            logger.error(
                "DeepSeek API exception on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
            
            # 检查是否为可重试的错误
            error_str = str(exc).lower()
            if any(error in error_str for error in ['rate limit', 'timeout', 'connection', 'server']) and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning("Retrying after %s seconds...", delay)
                time.sleep(delay)
                continue
            
            # 如果是认证错误或参数错误，不重试
            if any(error in error_str for error in ['authentication', 'invalid', 'parameter']):
                logger.error("Non-retryable error: %s", exc)
                break
                
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
                continue
                
            raise
    
    return None


def _log_usage_deepseek(response: Any) -> None:
    """
    记录 DeepSeek API 用量和成本估算
    """
    try:
        usage = response.usage
        if not usage:
            return
            
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens or 0
        
        # 计算成本 (元)
        cost = (prompt_tokens / 1000 * PRICE_INPUT_PER_1K) + (
            completion_tokens / 1000 * PRICE_OUTPUT_PER_1K
        )
        
        logger.info(
            "DeepSeek API用量: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s, 估算价格=¥%.4f",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
        )
    except Exception as exc:
        logger.warning("无法解析DeepSeek API用量: %s", exc)


__all__ = ["create_deepseek_client", "post_with_retries_deepseek"]