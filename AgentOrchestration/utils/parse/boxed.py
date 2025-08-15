

import re



def parse_boxed(content: str) -> str:
    """
    Parse the boxed content from the message.
    """
    match = re.search(r'\\boxed\{([^}]*)\}', content)
    if match:
        return match.group(1).strip()
    else:
        return content.strip()

