module rk.tautodiff.aux.tape;

import rk.tautodiff.core.common;
import rk.tautodiff.core.value;

/// Create a tape of equations and only update the values (is meant to be used in loops)
class Tape 
{
    Value[] values;
    alias lastValue this;
    private bool locked;

    /// Create Tape
    this()
    {
        locked = false;
    }

    /// Retrive the last pushed value
    Value lastValue()
    {
        return values.length ? values[$-1] : null;
    }

    /// Frees tape elements (resets the tape)
    void reset() {
        values.length = 0;
        locked = false;
    }

    /// Update tape elements values starting from the begining of the tape
    void update() {
        foreach (v; values) v.update();
    }

    /// Locks the tape. Tape can only be unlocked with `reset()` afterwards.
    void lock() {
        this.locked = true;
    }

    /// Checks if tape is locked
    bool isLocked() {
        return this.locked;
    }
}

