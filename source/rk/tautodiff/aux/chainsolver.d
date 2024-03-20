module rk.tautodiff.aux.chainsolver;

import rk.tautodiff.core.common;
import rk.tautodiff.core.value;

/// Solve equations
struct ChainSolver
{
    int step;
    Value[] values;
    alias lastResult this;

    /// Create ChainSolver with initial value
    this(in ElementType initialValue) 
    {
        step++;
        values ~= value(initialValue);
    }

    /// Reset operations
    void reset() 
    {
        step = 1;
        this.lastResult().zeroGrad();
    }

    /// Returns last result of calculation
    Value lastResult()
    {
        return values[step-1];
    }

    void opOpAssign(string op)(Value rhs) 
    {
        static if (op == "+" || op == "-")
        {
            if (step < values.length) values[step].reinit(
                mixin("this.lastResult.data" ~ op ~ "rhs.data"), 
                [this.lastResult, rhs], 
                &opBackwardAddSub
            );
            else values ~= value(
                mixin("this.lastResult.data" ~ op ~ "rhs.data"), 
                [this.lastResult, rhs], 
                &opBackwardAddSub
            );
        }
        else static if (op == "*" || op == "/")
        {
            if (step < values.length) values[step].reinit(
                mixin("this.lastResult.data" ~ op ~ "rhs.data"), 
                [this.lastResult, rhs], 
                &opBackwardMulDiv
            );
            else values ~= value(
                mixin("this.lastResult.data" ~ op ~ "rhs.data"), 
                [this.lastResult, rhs], 
                &opBackwardMulDiv
            );
        }
        else static if (op == "~")
        {
            if (step < values.length) values[step] = rhs;
            else values ~= rhs;
        }
        else static assert(0, "Operator <"~op~"> not supported!");

        // increment
        step++;
    }

    void opOpAssign(string op)(in ElementType rhs)
    {
        opOpAssign!op(value(rhs));
    }
}

unittest
{
    import std.stdio;

    // create solver
    auto solver = ChainSolver(0);

    /*
        OPERATIONS:
    */

    solver += 5; // 0 + 5 = 5
    solver -= 2; // 5 - 2 = 3
    solver *= 2; // 3 * 2 = 6
    solver /= 3; // 6 / 3 = 2
    assert(solver.data == 2);
    assert(solver.grad == 0);

    // backward
    solver.backward();
    assert(solver.grad == 1);

    // // zero grad
    solver.zeroGrad();
    assert(solver.grad == 0);

    /*
        APPEND ELEMENTS:
    */

    solver ~= 5.value * 13.value;
    assert(solver.data == 65);

    solver ~= 3.value  - 2.value * 3.value;
    assert(solver.data == -3);

    solver ~= 245.value;
    assert(solver.data == 245);
    assert(solver.values.length == 8);

    /*
        RESET:
    */

    solver.reset();
    assert(solver.data == 0);
    assert(solver.grad == 0);

    solver += 5;
    solver *= 2;
    solver -= 9;
    assert(solver.data == 1);
    assert(solver.grad == 0);
    assert(solver.values.length == 8);
}


