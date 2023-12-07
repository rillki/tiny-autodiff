module rk.tgrad.value;

import std.traits : isFloatingPoint;
import std.container.rbtree : RedBlackTree;

alias value = Value!float;

struct Value(T = float) if (isFloatingPoint!T)
{
    T data;
    T grad;
    typeof(this)[] children;
    void delegate() backward_fn;

    alias data this;

    this(in T data) 
    {
        this.data = data;
        this.grad = 0;
    }

    this(in T data, typeof(this)[] children) 
    {
        this(data);
        this.children = children;
    }

    typeof(this) opBinary(string op)(typeof(this) rhs) 
    {
        auto result = Value!T(mixin("this.data" ~ op ~ "rhs.data"), [this, rhs]);
        result.backward_fn = () {
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
        import std.algorithm : canFind, sort, uniq;
        
        typeof(this)[] topo;
        typeof(this)[] visited;
        void buildTopoTree(typeof(this) v) 
        {
            if (!visited.canFind(v)) 
            {
                visited ~= v;
                visited = visited.sort.uniq.array;
                foreach (typeof(this) child; v.children)
                {
                    buildTopoTree(child);
                }
                topo ~= v;
            }
        }

        // build tree recursively 
        buildTopoTree(this);

        // backward
        foreach_reverse (typeof(this) v; topo) {
            v.backward_fn();
        }
    }
}




