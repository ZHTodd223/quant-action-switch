# Experiment workspace policy

This directory is reserved for future versioned experiment workspaces. It is
empty during repository consolidation: no experiment has been started here.

Before a workspace is created, it must be registered in
`configs/experiment_registry.json`, have an approved locked configuration and
manifest, and state whether it is pilot or formal. Raw outputs and metrics are
immutable once recorded; corrections are sidecars. This consolidation creates
no scientific result.
