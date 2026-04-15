from __future__ import annotations

import os

from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app

try:
    from onnx_env.models import OnnxAction, OnnxObservation
    from onnx_env.server.onnx_env_environment import OnnxEnvironment
except ModuleNotFoundError:
    from models import OnnxAction, OnnxObservation  # type: ignore[import-not-found]
    from server.onnx_env_environment import OnnxEnvironment  # type: ignore[import-not-found]

MAX_CONCURRENT_ENVS = int(os.environ.get("MAX_CONCURRENT_ENVS", "32"))
ENABLE_WEB_INTERFACE = os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() == "true"
ENV_NAME = "onnx_deployment_surgeon_gym"


def create_environment() -> OnnxEnvironment:
    return OnnxEnvironment()


if ENABLE_WEB_INTERFACE:
    try:
        from openenv.core.env_server import create_web_interface_app

        app = create_web_interface_app(
            create_environment,
            OnnxAction,
            OnnxObservation,
            env_name=ENV_NAME,
            max_concurrent_envs=MAX_CONCURRENT_ENVS,
        )
    except (ModuleNotFoundError, ImportError):
        ENABLE_WEB_INTERFACE = False

if not ENABLE_WEB_INTERFACE:
    app = create_app(
        create_environment,
        OnnxAction,
        OnnxObservation,
        env_name=ENV_NAME,
        max_concurrent_envs=MAX_CONCURRENT_ENVS,
    )


@app.get("/manifest.json", include_in_schema=False)
async def web_manifest():
    return JSONResponse(
        {
            "name": "ONNX Deployment Surgeon Gym",
            "short_name": "OnnxSurgeon",
            "description": "Repair ONNX deployment incidents across mobile, browser, packaging, quantized, and release profiles.",
            "start_url": "/web/",
            "display": "standalone",
            "background_color": "#0f172a",
            "theme_color": "#059669",
            "icons": [
                {
                    "src": "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
                    "sizes": "any",
                    "type": "image/svg+xml",
                }
            ],
        }
    )


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    return JSONResponse({})


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
