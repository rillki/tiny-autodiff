module rk.tautodiff.aux.chainsolver;

import rk.tautodiff.core.common;
import rk.tautodiff.core.value;

/// Reuse Values[] for iterative solves
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
        step = 0;
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
                mixin("this.values[$-1].data" ~ op ~ "rhs.data"), 
                [this.values[$-1], rhs], 
                &opBackwardAddSub
            );
            else values ~= value(
                mixin("this.values[$-1].data" ~ op ~ "rhs.data"), 
                [this.values[$-1], rhs], 
                &opBackwardAddSub
            );
        }
        else static if (op == "*" || op == "/")
        {
            if (step < values.length) values[step].reinit(
                mixin("this.values[$-1].data" ~ op ~ "rhs.data"), 
                [this.values[$-1], rhs], 
                &opBackwardMulDiv
            );
            else values ~= value(
                mixin("this.values[$-1].data" ~ op ~ "rhs.data"), 
                [this.values[$-1], rhs], 
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

    solver.values.writeln();
    solver.writeln();
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
}


