import qsdsan as qs
import numpy as np
from exposan import bsm1 # Used for loading a system with pre-defined units/streams/components

# --- Example of setting up a simple model ---
# Ensure a system is loaded or created to have units/streams/components registered.
# bsm1.load() initializes the BSM1 system and registers its units, streams, and components
# in the default flowsheet.
bsm1.load()
sys = bsm1.sys

print("Initial setup complete. System units (first 3):")
# CORRECTED LINE: Directly convert the unit registry to a list
print(list(qs.Flowsheet.flowsheet.default.unit)[:3])

print("\nInitial setup complete. Components (first 5 IDs):")
# qs.get_thermo().chemicals is the CompiledComponents object
print(qs.get_thermo().chemicals.IDs[:5])

# --- Clear everything ---
print("\nClearing QSDsan registries...")
# CORRECTED LINE: Call clear() on the default flowsheet instance
qs.Flowsheet.flowsheet.default.clear()

# --- Verify that registries are empty ---
print("\nAfter clearing - System units:")
# Check if the unit registry is empty. It should show an empty registry.
print(qs.Flowsheet.flowsheet.default.unit)

print("After clearing - Components:")
# Check if the components registry is empty. It should show CompiledComponents([]).
print(qs.get_thermo().chemicals)

# Now you can redefine your components, units, and system as if starting fresh
# without worrying about old definitions