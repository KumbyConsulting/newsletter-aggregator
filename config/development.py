"""Development configuration."""

DEBUG = True
TEMPLATES_AUTO_RELOAD = True
SEND_FILE_MAX_AGE_DEFAULT = 0
DEBUG_TB_INTERCEPT_REDIRECTS = False
DEBUG_TB_TEMPLATE_EDITOR_ENABLED = True

# Debug toolbar configuration
DEBUG_TB_ENABLED = True
DEBUG_TB_PROFILER_ENABLED = True
DEBUG_TB_PANELS = [
    'flask_debugtoolbar.panels.versions.VersionDebugPanel',
    'flask_debugtoolbar.panels.timer.TimerDebugPanel',
    'flask_debugtoolbar.panels.headers.HeaderDebugPanel',
    'flask_debugtoolbar.panels.request_vars.RequestVarsDebugPanel',
    'flask_debugtoolbar.panels.config_vars.ConfigVarsDebugPanel',
    'flask_debugtoolbar.panels.template.TemplateDebugPanel',
    'flask_debugtoolbar.panels.logger.LoggingPanel',
    'flask_debugtoolbar.panels.route_list.RouteListDebugPanel',
    'flask_debugtoolbar.panels.profiler.ProfilerDebugPanel',
] 