"""Project settings.

Configured lazily so that importing the package for the API or Dash
does not trigger the Kedro config loader.
"""

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        settings.configure(
            config_loader=OmegaConfigLoader(conf_source="conf"),
            package_name="retail_demand_forecasting",
            project_name="retail_demand_forecasting",
        )
        _configured = True
