# Physical robot boundary

This domain owns robot adapters and sensor/control clients. SDK processes and
optional hardware dependencies live in independently installed `integrations`
packages. Go2 hardware is owned by `integrations/go2_owner`, outside the Host.

`robots/mobile_base` defines shared motion, observation and navigation contracts
used by Go2, simulators and `application/navigation`. Go2-specific HTTP clients
and adaptation remain in `robots/unitree/go2`.

Model communication belongs to `model_services`; policy-to-robot translation
belongs to `bindings`; navigation execution loops belong to `application/navigation`.
Hardware deployment belongs to `deployment/go2`, independent of model choice.
