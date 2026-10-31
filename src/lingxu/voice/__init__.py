"""Discord DAVE formal Meeting and resident-assistant voice primitives.

Concrete modules are intentionally not imported here: the formal AudioSink composes
the realtime-ASR service, whose queues in turn consume voice frame value objects.
Keeping the package initializer side-effect free prevents that valid dependency
direction from becoming an import cycle.
"""
