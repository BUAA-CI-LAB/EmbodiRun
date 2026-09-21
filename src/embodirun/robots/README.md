# Physical robot boundary

This domain owns vendor SDK bindings, sensor and control clients, and the
small services deployed on a robot. It implements task-facing interfaces such
as `robots.unitree.go2.navigation.MobileBase` and `ObservationSource`.

It does not load models or translate model-native outputs. Model semantics
belong to `models`; translating those outputs into task plans belongs to
`policies`; closed-loop execution belongs to `tasks`.
