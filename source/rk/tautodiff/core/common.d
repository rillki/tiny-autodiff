module rk.tautodiff.core.common;

version(TAUTODIFF_USE_DOUBLE) alias ElementType = double;
else version(TAUTODIFF_USE_REAL) alias ElementType = real;
else alias ElementType = float;

