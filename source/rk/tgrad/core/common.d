module rk.tgrad.core.common;

version(TGRAD_USE_DOUBLE) alias ElementType = double;
else version(TGRAD_USE_REAL) alias ElementType = real;
else alias ElementType = float;

