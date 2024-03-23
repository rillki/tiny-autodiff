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
        foreach (ref v; values) v.update();
    }

    /// Locks the tape. Tape can only be unlocked with `reset()` afterwards.
    void lock() {
        this.locked = true;
    }

    /// Checks if tape is locked
    bool isLocked() {
        return this.locked;
    }

    // Push back a single value
    void pushBack(Value v) 
    {
        this.values ~= v;
    }

    /// Push back values
    void pushBack(Value[] vs) 
    {
        this.values ~= vs;
    }

    /// Append a single value
    void opOpAssign(string op: "~")(Value v)
    {
        this.pushBack(v);
    }

    /// Append values
    void opOpAssign(string op: "~")(Value[] vs)
    {
        this.pushBack(vs);
    }
}

unittest
{
    // init
    auto tape = new Tape();
    assert(tape.values == []);
    assert(tape.values.length == 0);
    assert(tape.locked == false);
    assert(!tape.isLocked);

    // d = a * b - c
    auto a = 5.value;
    auto b = 10.value;
    auto c = 25.value;
    auto d = a * b;
    auto e = d - c;
    assert(e.data == 25);

    // push
    tape.pushBack(a);
    tape ~= b;
    tape ~= [c, d, e];
    assert(tape.values == [a, b, c, d, e]);
    assert(tape.values.length == 5);
    assert(tape.lastValue.data == 25);

    // modify value
    a.data = 6;
    
    // update tape
    tape.update();
    assert(tape.lastValue.data == 35);
}
