module rk.tgrad.value;

import std.traits : isFloatingPoint;
import std.container.rbtree : RedBlackTree;

alias value = Value!float;

struct Value(T = float) if (isFloatingPoint!T)
{
    T data;
    T grad;
    typeof(this)[] children;
    void delegate() _backward;
    size_t backwardCount;

    alias data this;

    this(in T data) 
    {
        this.data = data;
        this.grad = 0;
        this.backwardCount = 0;
    }

    this(in T data, typeof(this)[] children) 
    {
        this(data);
        this.children = children;
    }

    typeof(this) opBinary(string op)(typeof(this) rhs) 
    {
        auto result = Value!T(mixin("this.data" ~ op ~ "rhs.data"), [this, rhs]);
        result.grad = 1;
        result._backward = () {
            static if (op == "+" || op == "-") {
                this.grad += result.grad;
                rhs.grad += result.grad;
            } else static if (op == "*" || op == "/") {
                this.grad += mixin("rhs.data" ~ op ~ "result.grad");
                rhs.grad += mixin("this.data" ~ op ~ "result.grad");
            } else static assert(0, "Operator <"~op~"> not supported!");
        };
        return result;
    }

    void backward() {
        import std.array : array;
        import std.algorithm : sort, uniq, canFind;

        // build topo list
        typeof(this)[] topoList = [];
        void buildTopoList(typeof(this) visitor) 
        {
            if (!topoList.canFind(visitor)) topoList ~= visitor;
            foreach (v; visitor.children)
            {   
                buildTopoList(v);
            }
        }
        buildTopoList(this);

        // perform backward propagation
        // grad = 1;
        foreach (child; topoList)
        {   
            if (child._backward) child._backward();
        }

        // grad = 1;
        // foreach (typeof(this) child; topoList)
        // {   
            
        // }        
        // grad = 1;
        // if (_backward) _backward();
        // foreach (typeof(this) child; children)
        // {   
        //     if (child.backwardCount == 0) child.backward();
        // }
        // backwardCount++;
    }

    void resetGrads() {
        grad = 0;
        backwardCount = 0;
    }
}




