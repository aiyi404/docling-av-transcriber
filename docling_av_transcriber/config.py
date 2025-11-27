import json
import os
from dataclasses import dataclass, field


@dataclass
class AliyunBailianSettings:
    """Configuration for Aliyun Bailian / DashScope ASR."""

    api_key: str | None = None
    # 默认北京地域，可按需改为 https://dashscope-intl.aliyuncs.com/api/v1
    base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1"
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/generation"
    file_upload_endpoint: str | None = None
    model: str = "fun-asr"
    timeout: int = 120
    retries: int = 3
    language: str = "zh"
    enable_words: bool = True
    diarization: bool = False
    # 按顺序尝试这些环境变量，兼容 ALIYUN_BAILIAN_API_KEY 与官方示例的 DASHSCOPE_API_KEY
    api_key_envs: tuple[str, ...] = field(
        default_factory=lambda: ("ALIYUN_BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
    )

    @classmethod
    def from_env(cls) -> "AliyunBailianSettings":
        default_envs: tuple[str, ...] = ("ALIYUN_BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
        env_field = cls.__dataclass_fields__.get("api_key_envs")
        if env_field and getattr(env_field, "default_factory", None) is not None:  # type: ignore[attr-defined]
            default_envs = env_field.default_factory()  # type: ignore[misc]

        api_key = None
        for env_name in default_envs:
            api_key = os.getenv(env_name)
            if api_key:
                break

        base_http_api_url = os.getenv("DASHSCOPE_BASE_HTTP_API_URL", cls.base_http_api_url)
        file_upload_endpoint = os.getenv(
            "DASHSCOPE_FILE_UPLOAD_ENDPOINT", f"{base_http_api_url.rstrip('/')}/files"
        )

        return cls(
            api_key=api_key,
            base_http_api_url=base_http_api_url,
            file_upload_endpoint=file_upload_endpoint,
        )


@dataclass
class AliyunVisionSettings:
    """Configuration for Aliyun DashScope multimodal vision."""

    api_key: str | None = None
    endpoint: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    )
    model: str = "qwen-vl-max"
    prompt: str = "请用中文详细描述这张关键帧的场景、主体、动作、文字以及潜在含义。"
    timeout: int = 120
    retries: int = 3
    parameters: dict[str, object] = field(default_factory=dict)
    api_key_envs: tuple[str, ...] = field(
        default_factory=lambda: ("ALIYUN_BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
    )

    @classmethod
    def from_env(cls) -> "AliyunVisionSettings":
        default_envs: tuple[str, ...] = ("ALIYUN_BAILIAN_API_KEY", "DASHSCOPE_API_KEY")
        env_field = cls.__dataclass_fields__.get("api_key_envs")
        if env_field and getattr(env_field, "default_factory", None) is not None:  # type: ignore[attr-defined]
            default_envs = env_field.default_factory()  # type: ignore[misc]

        api_key = None
        for env_name in default_envs:
            api_key = os.getenv(env_name)
            if api_key:
                break

        endpoint = os.getenv("ALIYUN_VISION_ENDPOINT", cls.endpoint)
        model = os.getenv("ALIYUN_VISION_MODEL", cls.model)
        prompt = os.getenv("ALIYUN_VISION_PROMPT", cls.prompt)
        timeout = int(os.getenv("ALIYUN_VISION_TIMEOUT", cls.timeout))
        retries = int(os.getenv("ALIYUN_VISION_RETRIES", cls.retries))

        parameters_env = os.getenv("ALIYUN_VISION_PARAMETERS")
        parameters: dict[str, object] = {}
        if parameters_env:
            try:
                parsed = json.loads(parameters_env)
                if isinstance(parsed, dict):
                    parameters = parsed
            except json.JSONDecodeError:
                pass

        return cls(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            timeout=timeout,
            retries=retries,
            parameters=parameters,
        )
