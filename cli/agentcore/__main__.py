"""Allow ``python -m cli.agentcore`` invocation."""

import sys

from .sync import main

sys.exit(main())
