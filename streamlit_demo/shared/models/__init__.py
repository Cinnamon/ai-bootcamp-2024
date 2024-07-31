from .engine import (
    engine
)
from .models import (
    Feedback, Base
)

Base.metadata.create_all(engine)
