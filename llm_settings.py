"""Read an explicitly selected local LLM profile without mixing providers."""
import ast
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def load_llm_settings(config_path=None):
    path = Path(config_path) if config_path is not None else Path(__file__).with_name('llm_config.py')
    values = {}
    if path.exists():
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                        'API_KEY', 'API_URL', 'BASE_URL', 'MODEL', 'MODEL_NAME',
                        'LLM_PROVIDER', 'Silicon_API_KEY', 'Silicon_API_URL',
                        'Silicon_BASE_URL', 'Silicon_MODEL', 'Silicon_MODEL_NAME',
                    ):
                        value = ast.literal_eval(node.value)
                        if not isinstance(value, str):
                            raise ValueError('LLM settings must be strings')
                        values[target.id] = value
        except (SyntaxError, ValueError, TypeError):
            # Do not include offending source lines: they may contain a key.
            raise ValueError('Invalid LLM config: {}'.format(path)) from None
    provider = values.get('LLM_PROVIDER', 'default').strip().lower()
    if provider not in ('default', 'silicon', 'siliconflow'):
        raise ValueError('Unknown LLM_PROVIDER; use default or silicon')
    prefix = '' if provider == 'default' else 'Silicon_'
    url = values.get(prefix + 'API_URL') or values.get(prefix + 'BASE_URL', '')
    if url and not values.get(prefix + 'API_URL'):
        parts = urlsplit(url)
        endpoint = parts.path.rstrip('/')
        if not endpoint.endswith(('/responses', '/chat/completions')):
            endpoint += '/chat/completions'
        url = urlunsplit((parts.scheme, parts.netloc, endpoint, parts.query, parts.fragment))
    settings = {
        'api_key': values.get(prefix + 'API_KEY', ''),
        'api_url': url,
        'model': values.get(prefix + 'MODEL_NAME') or values.get(prefix + 'MODEL') or ('Qwen/Qwen3-8B' if provider == 'default' else ''),
    }
    if provider == 'default':
        for field, variable in (('api_key', 'LLM_API_KEY'), ('api_url', 'LLM_API_URL'), ('model', 'LLM_MODEL_NAME')):
            settings[field] = os.environ.get(variable, settings[field])
    elif not all(settings.values()):
        raise ValueError('Silicon profile requires a key, URL and model')
    return settings
