module rk.tgrad.value;

import std.traits : isFloatingPoint;
import std.container.rbtree : RedBlackTree;

struct Value(T = float) if (isFloatingPoint!T)
{
    T data;
    T grad;
    typeof(this)[] children;
    void delegate(ref typeof(this) grads) backward;

    alias data this;

    this(in T data) {
        this.data = data;
        this.grad = 0;
    }
}




