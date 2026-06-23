"""Import all algorithm modules to trigger @register_algorithm decorators.

Adding a new algorithm
----------------------
1. Create or edit the appropriate module (cdom.py, chla.py, spm.py, turbidity.py).
2. Decorate your class with @register_algorithm('product', 'name').
3. That's all — no changes needed anywhere else.
"""

from . import cdom       # noqa: F401
from . import chla       # noqa: F401
from . import spm        # noqa: F401
from . import turbidity  # noqa: F401
