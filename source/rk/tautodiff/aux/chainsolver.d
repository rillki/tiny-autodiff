module rk.tautodiff.aux.chainsolver;

import rk.tautodiff.core.common;
import rk.tautodiff.core.value;

/// Reuse Values[] for iterative solves
struct ChainSolver
{
    int step;
    Value[] values;

    /// Reset operations
    void reset() 
    {
        step = 0;
    }
}

