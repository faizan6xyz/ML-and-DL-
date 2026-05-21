"""
================================================================================
  TENSORFLOW GRAPH MODE — COMPLETE GUIDE
  Topics: Static Graphs, tf.function, AutoGraph, Concrete Functions,
          Graph Tracing, tf.Module, Custom Ops, Graph Profiling,
          SavedModel Graph Export, Graph Optimization & Execution
================================================================================

  WHY GRAPHS?
  ──────────────────────────────────────────────────────────────────────────────
  TensorFlow has two execution modes:

  1. EAGER MODE  (default since TF 2.x)
     • Operations run immediately, Python-style.
     • Easy to debug, but slower — Python overhead on every op.
     • USE CASE: prototyping, debugging, interactive exploration.

  2. GRAPH MODE  (tf.function, TF 1.x sessions)
     • Operations are compiled into a static dataflow graph (GraphDef).
     • Graph is optimized by XLA / Grappler before execution.
     • USE CASE: production training, inference, deployment, edge devices.

  GRAPH ADVANTAGES:
     ✓ 2–10× faster execution (Python overhead eliminated)
     ✓ Portable — graphs run on CPU / GPU / TPU / mobile / edge
     ✓ Serializable — save and restore full computation
     ✓ Parallelizable — TF schedules independent ops concurrently
     ✓ Optimizable — dead-code elimination, op fusion, constant folding
================================================================================
"""

import time, inspect, numpy as np
import tensorflow as tf

print(f"TensorFlow : {tf.__version__}")
print(f"Eager mode : {tf.executing_eagerly()}")  # True in TF2 by default


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EAGER vs GRAPH: THE FUNDAMENTAL DIFFERENCE
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Understanding when Python code runs vs when the graph runs.
#            Critical for debugging side-effects (print, assertions, etc.)

def eager_demo():
    """Every line executes immediately. Results are Python values."""
    a = tf.constant([1.0, 2.0, 3.0])   # creates tensor NOW
    b = tf.constant([4.0, 5.0, 6.0])
    c = a + b                            # adds NOW → tf.Tensor([5,7,9])
    print(f"Eager result: {c.numpy()}")  # .numpy() works immediately
    return c

eager_demo()


@tf.function                             # ← compiles to a graph
def graph_demo(a, b):
    """
    On FIRST call  → Python body is TRACED (executed once to build graph).
    On ALL calls   → only the graph executes (Python body is NOT re-run).

    IMPORTANT: print() runs only during tracing, NOT during every call.
    Use tf.print() for runtime logging inside graphs.
    """
    print("⚡ [PYTHON] Tracing the graph...")   # runs once (during trace)
    tf.print("🔥 [TF]     Runtime execution")    # runs every call
    return a + b

print("\n─── First call (triggers tracing) ───")
graph_demo(tf.constant([1.0]), tf.constant([2.0]))

print("\n─── Second call (graph reused, no trace) ───")
graph_demo(tf.constant([3.0]), tf.constant([4.0]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — tf.function DEEP DIVE: TRACING, SIGNATURES, RETRACING
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Controlling when TF retraces to avoid performance pitfalls.
#            Each retrace = rebuilding the graph = slow startup.

@tf.function
def flexible_fn(x):
    """
    TF traces a NEW concrete function for each unique input signature:
      • Different dtype  → new trace
      • Different shape  → new trace (if shape not specified in input_signature)
      • Different rank   → new trace
    Call .pretty_printed_concrete_signatures() to inspect all traces.
    """
    return tf.square(x) + tf.abs(x)

# Three different signatures → three traces
flexible_fn(tf.constant(2.0))          # float32 scalar
flexible_fn(tf.constant([1, 2, 3]))    # int32 vector
flexible_fn(tf.constant([[1.0, 2.0]])) # float32 matrix

print(f"\nConcrete functions created: "
      f"{len(flexible_fn._list_all_concrete_functions_for_serialization())}")


# ── Fixing the input signature prevents unwanted retracing ───────────────────
# USE CASE: Production APIs where input shapes/dtypes are known ahead of time.

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 20], dtype=tf.float32, name="features"),
    tf.TensorSpec(shape=[None],     dtype=tf.int32,   name="labels")
])
def fixed_signature_fn(features, labels):
    """
    input_signature freezes the graph to ONE concrete function.
    Any input that matches the spec reuses this single graph.
    Benefit: zero retracing overhead, fully serializable.
    """
    labels_one_hot = tf.one_hot(labels, depth=5)
    logits         = tf.linalg.matmul(features,
                                       tf.ones([20, 5], dtype=tf.float32))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels_one_hot, logits))
    return {"logits": logits, "loss": loss}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AUTOGRAPH: PYTHON CONTROL FLOW → GRAPH OPS
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Writing readable Python (if/for/while/break/return) that TF
#            automatically converts to graph ops (tf.cond / tf.while_loop).

@tf.function
def autograph_conditionals(x):
    """
    AutoGraph rewrites this Python if/else into tf.cond.
    RULE: the condition MUST be a tf.Tensor, not a Python bool,
          for the branch to become a graph op (not a trace-time branch).
    """
    if x > 0:                      # → tf.cond at graph level
        return tf.math.log(x)
    else:
        return tf.math.exp(x)

# Inspect the generated graph code
print("\n─── AutoGraph generated source ───")
print(tf.autograph.to_code(autograph_conditionals.python_function))


@tf.function
def autograph_while_loop(n):
    """
    Python while loop over a tf.Tensor → tf.while_loop in the graph.
    USE CASE: Dynamic-length sequences, iterative algorithms (e.g. power method).
    """
    i   = tf.constant(0)
    acc = tf.constant(0.0)
    while i < n:                   # → tf.while_loop
        acc += tf.cast(i, tf.float32) ** 2
        i   += 1
    return acc

result = autograph_while_loop(tf.constant(10))
print(f"\nSum of squares 0..9 = {result.numpy()}")   # 285.0


@tf.function
def autograph_for_loop(items):
    """
    Python for loop over a tf.Tensor → tf.while_loop.
    USE CASE: Processing variable-length tensors without Python overhead.
    """
    total = tf.constant(0.0)
    for item in items:             # → tf.while_loop
        total += item
    return total

print(f"Sum = {autograph_for_loop(tf.constant([1.0, 2.0, 3.0, 4.0])).numpy()}")


# ── When NOT to use AutoGraph ─────────────────────────────────────────────────
# USE CASE: Debugging — temporarily disable graph compilation.

@tf.function(experimental_autograph_options=
             tf.autograph.experimental.Feature.NONE)
def no_autograph(x):
    """
    Disables AutoGraph. Python if/for run at TRACE TIME only.
    Use when: the logic depends only on static Python values,
              or when you want to debug raw eager behaviour.
    """
    if isinstance(x, int):         # trace-time Python check (ok here)
        return tf.constant(x * 2)
    return tf.square(x)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CONCRETE FUNCTIONS & GET_CONCRETE_FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Inspecting, exporting, and directly calling a single compiled
#            graph for maximum performance and portability.

@tf.function
def matmul_add(a, b, c):
    return tf.linalg.matmul(a, b) + c

# Obtain a concrete function for a fixed shape
concrete = matmul_add.get_concrete_function(
    tf.TensorSpec([4, 8],  tf.float32),
    tf.TensorSpec([8, 16], tf.float32),
    tf.TensorSpec([4, 16], tf.float32),
)

print(f"\nConcrete function name  : {concrete.name}")
print(f"Structured input specs  : {concrete.structured_input_signature}")
print(f"Structured output specs : {concrete.structured_outputs}")

# Call the concrete function directly (skips dispatch overhead)
a = tf.random.normal([4, 8])
b = tf.random.normal([8, 16])
c = tf.random.normal([4, 16])
out = concrete(a, b, c)
print(f"Output shape: {out.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GRAPH INTROSPECTION: NODES, EDGES, OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Debugging graph structure, verifying op placement,
#            optimizing by identifying redundant nodes.

@tf.function
def simple_graph(x):
    y = tf.square(x)
    z = tf.sqrt(tf.abs(y))
    return z + tf.constant(1.0)

cf = simple_graph.get_concrete_function(tf.TensorSpec([], tf.float32))
graph = cf.graph

print(f"\n─── Graph Operations ({len(list(graph.get_operations()))}) ───")
for op in graph.get_operations():
    inputs  = [i.name for i in op.inputs]
    outputs = [o.name for o in op.outputs]
    print(f"  [{op.type:20s}]  {op.name}")
    if inputs:
        print(f"     inputs : {inputs}")

# Access individual tensors in the graph
print(f"\nGraph inputs  : {[i.name for i in cf.inputs]}")
print(f"Graph outputs : {[o.name for o in cf.outputs]}")

# Check which device an op is placed on
for op in list(graph.get_operations())[:3]:
    print(f"  {op.name} → device: '{op.device or 'default'}'")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VARIABLES IN GRAPHS: tf.Variable LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Model weights, optimizer states, running statistics.
#            Variables persist across tf.function calls; tensors do not.

# RULE: Create tf.Variable OUTSIDE @tf.function (at Python level).
#       Creating inside @tf.function raises an error on the 2nd trace.

counter = tf.Variable(0, dtype=tf.int32, name="step_counter",
                       trainable=False)
weights = tf.Variable(tf.random.normal([10, 5]), name="dense_weights")

@tf.function
def update_variables(x):
    """
    tf.Variable.assign()       → set value in graph
    tf.Variable.assign_add()   → increment in graph
    tf.Variable.assign_sub()   → decrement in graph
    These return an op; the update is part of the graph execution.
    """
    counter.assign_add(1)
    new_weights = weights + x
    weights.assign(new_weights)        # in-place update (no new allocation)
    return weights, counter

for _ in range(3):
    w, c = update_variables(tf.ones([10, 5]) * 0.1)

print(f"\nCounter after 3 steps: {counter.numpy()}")
print(f"Weights mean         : {tf.reduce_mean(weights).numpy():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — tf.Module: REUSABLE, SERIALIZABLE GRAPH COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Building modular, portable computation blocks that can be saved
#            independently and composed into larger systems.

class LinearLayer(tf.Module):
    """
    tf.Module automatically tracks tf.Variable attributes.
    Use .trainable_variables / .variables to inspect all tracked state.
    """
    def __init__(self, in_features, out_features, name="linear"):
        super().__init__(name=name)
        self.W = tf.Variable(
            tf.initializers.GlorotUniform()([in_features, out_features]),
            name="weights")
        self.b = tf.Variable(tf.zeros([out_features]), name="bias")

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def __call__(self, x):
        return tf.linalg.matmul(x, self.W) + self.b


class MLP(tf.Module):
    """
    Composed module: stacks LinearLayers + activation.
    USE CASE: Custom architectures where Keras overhead is unwanted
               (e.g., research models, C++ serving).
    """
    def __init__(self, layer_sizes, name="mlp"):
        super().__init__(name=name)
        self.layers_list = []
        for i, (in_f, out_f) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers_list.append(LinearLayer(in_f, out_f, f"linear_{i}"))

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def __call__(self, x):
        for i, layer in enumerate(self.layers_list[:-1]):
            x = tf.nn.relu(layer(x))
        return self.layers_list[-1](x)   # no activation on final layer


mlp = MLP([16, 64, 32, 8])
dummy = tf.random.normal([4, 16])
out   = mlp(dummy)
print(f"\nMLP output shape: {out.shape}")
print(f"Total variables : {len(mlp.variables)}")
print(f"Trainable vars  : {len(mlp.trainable_variables)}")

# Save the tf.Module as SavedModel
tf.saved_model.save(mlp, "/tmp/mlp_module")
loaded_mlp = tf.saved_model.load("/tmp/mlp_module")
out_loaded  = loaded_mlp(dummy)
print(f"Loaded MLP output shape: {out_loaded.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GRADIENT TAPE IN GRAPH MODE
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Custom training loops, meta-learning (MAML),
#            second-order gradients, research-grade optimizers.

W = tf.Variable(tf.random.normal([4, 2]), name="W")
b = tf.Variable(tf.zeros([2]), name="b")

@tf.function
def compute_loss(X, y_true):
    logits = tf.linalg.matmul(X, W) + b
    loss   = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, logits))
    return loss, logits

@tf.function
def train_step_with_tape(X, y_true, lr=0.01):
    """
    GradientTape records forward-pass ops inside 'with' block.
    .gradient() differentiates loss w.r.t. watched variables.
    
    USE CASE: Any situation where model.fit() is too restrictive —
               GANs (two optimizers), RL (custom reward shaping),
               multi-task (weighted loss combination).
    """
    with tf.GradientTape() as tape:
        loss, logits = compute_loss(X, y_true)

    grads = tape.gradient(loss, [W, b])   # ∂loss/∂W, ∂loss/∂b
    W.assign_sub(lr * grads[0])           # gradient descent step
    b.assign_sub(lr * grads[1])
    return loss, logits


@tf.function
def higher_order_gradients(x):
    """
    Nested GradientTapes compute Hessians / Jacobians.
    USE CASE: Second-order optimizers (Newton's method, L-BFGS),
               physics-informed neural networks (PINNs),
               neural ODEs.
    """
    with tf.GradientTape() as outer:
        outer.watch(x)
        with tf.GradientTape() as inner:
            inner.watch(x)
            y = tf.reduce_sum(x ** 3)
        grad1 = inner.gradient(y, x)      # ∂y/∂x = 3x²
    grad2 = outer.gradient(grad1, x)       # ∂²y/∂x² = 6x (Hessian diagonal)
    return grad1, grad2

x_val      = tf.constant([1.0, 2.0, 3.0])
g1, g2     = higher_order_gradients(x_val)
print(f"\n∂y/∂x  (should be 3x²)  : {g1.numpy()}")   # [3, 12, 27]
print(f"∂²y/∂x² (should be 6x) : {g2.numpy()}")   # [6, 12, 18]


@tf.function
def jacobian_example(x):
    """
    USE CASE: Sensitivity analysis, Jacobian-vector products,
               computing Fisher information matrices.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.stack([x[0]**2 + x[1],       # output 0
                      x[0] * x[1],            # output 1
                      x[1]**3])               # output 2
    return tape.jacobian(y, x)               # shape [3, 2]

x_jac  = tf.constant([2.0, 3.0])
J      = jacobian_example(x_jac)
print(f"\nJacobian:\n{J.numpy()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — GRAPH PERFORMANCE: BENCHMARKING EAGER vs GRAPH
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Quantifying the speed-up from @tf.function.
#            Helps justify graph compilation overhead for specific workloads.

def heavy_computation_eager(x):
    for _ in range(100):
        x = tf.linalg.matmul(x, tf.transpose(x))
        x = tf.nn.relu(x)
        x = x / (tf.reduce_max(x) + 1e-8)
    return x

@tf.function
def heavy_computation_graph(x):
    """
    Same computation compiled to graph.
    TF Grappler optimizer will:
      • Fuse compatible ops (e.g., matmul + bias_add → FusedMatMul)
      • Eliminate dead branches
      • Apply constant folding
      • Optimize memory layout
    """
    for _ in range(100):
        x = tf.linalg.matmul(x, tf.transpose(x))
        x = tf.nn.relu(x)
        x = x / (tf.reduce_max(x) + 1e-8)
    return x

mat = tf.random.normal([64, 64])

# Warmup
heavy_computation_eager(mat)
heavy_computation_graph(mat)

REPS = 10
t0 = time.perf_counter()
for _ in range(REPS):
    heavy_computation_eager(mat)
eager_time = (time.perf_counter() - t0) / REPS

t0 = time.perf_counter()
for _ in range(REPS):
    heavy_computation_graph(mat)
graph_time = (time.perf_counter() - t0) / REPS

print(f"\n─── Performance Benchmark ───")
print(f"Eager mode : {eager_time*1000:.2f} ms/call")
print(f"Graph mode : {graph_time*1000:.2f} ms/call")
print(f"Speed-up   : {eager_time/graph_time:.2f}×")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — tf.cond AND tf.switch_case: GRAPH-LEVEL BRANCHING
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Runtime branching inside graphs where the condition is a tensor.
#            Necessary for dynamic control flow in inference pipelines,
#            e.g., beam search, conditional computation, mixture-of-experts.

@tf.function
def graph_branching(x, training):
    """
    tf.cond evaluates BOTH branches during graph construction,
    but only EXECUTES one branch at runtime — critical distinction.
    
    USE CASE: Switching between train/eval behaviour inside a single graph
               (Dropout, BatchNorm, etc.)
    """
    def train_branch():
        noise = tf.random.normal(tf.shape(x), stddev=0.1)
        return x + noise          # add noise during training

    def eval_branch():
        return x                  # clean input during evaluation

    return tf.cond(training, train_branch, eval_branch)

x_in = tf.constant([1.0, 2.0, 3.0])
print(f"\nTraining output: {graph_branching(x_in, tf.constant(True)).numpy()}")
print(f"Eval output    : {graph_branching(x_in, tf.constant(False)).numpy()}")


@tf.function
def graph_switch_case(mode, x):
    """
    tf.switch_case: multi-way branching.
    USE CASE: Mixture-of-experts (route input to one of N expert networks),
               conditional computation based on token type, A/B testing.
    """
    def expert_0(): return tf.square(x)
    def expert_1(): return tf.sqrt(tf.abs(x))
    def expert_2(): return tf.math.log(x + 1.0)

    return tf.switch_case(mode, branch_fns=[expert_0, expert_1, expert_2])

for m in range(3):
    val = graph_switch_case(tf.constant(m), tf.constant(4.0))
    print(f"  Expert {m} output: {val.numpy():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — tf.while_loop: GRAPH-LEVEL ITERATION
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: RNN unrolling, iterative solvers, beam search, simulation loops
#            where the number of steps is determined at RUNTIME.

@tf.function
def power_iteration(A, num_iters=50):
    """
    Finds dominant eigenvector of matrix A via power iteration.
    USE CASE: PCA in production pipelines, spectral normalization for GANs,
               PageRank-style computations in graph neural networks.
    """
    n  = tf.shape(A)[0]
    v  = tf.random.normal([n, 1])
    v  = v / tf.norm(v)

    def body(i, v):
        v = tf.linalg.matmul(A, v)
        v = v / tf.norm(v)
        return i + 1, v

    def cond(i, v):
        return i < num_iters

    _, v_final = tf.while_loop(
        cond, body,
        loop_vars=[tf.constant(0), v],
        # shape_invariants: tell TF which dimensions can change
        shape_invariants=[tf.TensorSpec([], tf.int32),
                          tf.TensorSpec([None, 1], tf.float32)]
    )
    eigenvalue = tf.squeeze(
        tf.linalg.matmul(tf.transpose(v_final),
                          tf.linalg.matmul(A, v_final)))
    return eigenvalue, v_final

A = tf.constant([[4.0, 1.0], [2.0, 3.0]])
eig_val, eig_vec = power_iteration(A)
print(f"\nDominant eigenvalue : {eig_val.numpy():.4f}")  # ≈ 5.0
print(f"Eigenvector         : {eig_vec.numpy().flatten()}")


@tf.function
def dynamic_rnn_loop(inputs, W_h, W_x, b):
    """
    Manual RNN unrolling with tf.while_loop.
    USE CASE: Custom RNN cells, variable-length sequences,
               attention mechanisms that stop early (conditional halt).
    inputs shape: [T, batch, features]
    """
    T      = tf.shape(inputs)[0]
    batch  = tf.shape(inputs)[1]
    hidden = tf.shape(W_h)[0]

    h = tf.zeros([batch, hidden])
    i = tf.constant(0)

    def body(i, h):
        x_t = inputs[i]
        h   = tf.nn.tanh(
            tf.linalg.matmul(h, W_h) + tf.linalg.matmul(x_t, W_x) + b)
        return i + 1, h

    _, h_final = tf.while_loop(
        lambda i, h: i < T, body,
        loop_vars=[i, h],
        shape_invariants=[tf.TensorSpec([], tf.int32),
                          tf.TensorSpec([None, None], tf.float32)]
    )
    return h_final

T, B, F, H = 10, 4, 8, 16
inp   = tf.random.normal([T, B, F])
W_h   = tf.random.normal([H, H]) * 0.1
W_x   = tf.random.normal([F, H]) * 0.1
b_rnn = tf.zeros([H])
h_out = dynamic_rnn_loop(inp, W_h, W_x, b_rnn)
print(f"\nRNN final hidden state: {h_out.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — tf.TensorArray: DYNAMIC GRAPH-COMPATIBLE ARRAYS
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Collecting per-step outputs inside tf.while_loop / @tf.function
#            when the number of steps is not known at trace time.

@tf.function
def collect_with_tensor_array(inputs):
    """
    Python lists can't be written inside tf.while_loop.
    TensorArray is the graph-native dynamic accumulator.

    USE CASE:
      • Saving all hidden states of an RNN (not just the final one)
      • Collecting loss values across gradient accumulation steps
      • Beam search — storing all candidate sequences
    """
    T      = tf.shape(inputs)[0]
    # dynamic_size=True: array grows automatically
    ta     = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                             element_shape=[inputs.shape[-1]])
    i      = tf.constant(0)

    def body(i, ta):
        processed = tf.nn.relu(inputs[i]) * tf.cast(i + 1, tf.float32)
        ta        = ta.write(i, processed)
        return i + 1, ta

    _, ta_final = tf.while_loop(
        lambda i, ta: i < T, body, [i, ta])

    return ta_final.stack()   # convert TensorArray → regular Tensor

seq    = tf.random.normal([6, 4])
result = collect_with_tensor_array(seq)
print(f"\nTensorArray stacked: {result.shape}")  # [6, 4]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — GRAPH-COMPATIBLE DATA OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: All tensor manipulation that must run inside @tf.function.

@tf.function
def graph_tensor_ops(x):
    """
    Common tensor operations and their graph equivalents.
    All run as graph ops — no Python, no NumPy, no lists.
    """
    # ── Shape operations ──────────────────────────────────────────────────────
    static_shape  = x.shape          # known at trace time (TensorShape)
    dynamic_shape = tf.shape(x)      # evaluated at runtime (tf.Tensor)

    # ── Slicing & indexing ────────────────────────────────────────────────────
    first_row  = x[0]                # static if index is Python int
    last_col   = x[:, -1]
    block      = x[1:3, 1:3]

    # ── Reshaping ─────────────────────────────────────────────────────────────
    flat       = tf.reshape(x, [-1])          # -1 = infer
    transposed = tf.transpose(x)

    # ── Math ──────────────────────────────────────────────────────────────────
    row_sum    = tf.reduce_sum(x, axis=1)
    col_mean   = tf.reduce_mean(x, axis=0)
    softmax    = tf.nn.softmax(x, axis=-1)
    normed     = tf.linalg.norm(x, axis=-1, keepdims=True)

    # ── Scatter / gather ──────────────────────────────────────────────────────
    indices    = tf.constant([0, 2])
    gathered   = tf.gather(x, indices)        # select rows 0 and 2

    # ── Casting ───────────────────────────────────────────────────────────────
    x_int      = tf.cast(x * 100, tf.int32)
    x_float16  = tf.cast(x, tf.float16)

    # ── Concatenation & stacking ──────────────────────────────────────────────
    doubled    = tf.concat([x, x], axis=0)    # [2N, M]
    stacked    = tf.stack([row_sum, row_sum])  # [2, M]

    return {
        "static_shape":  static_shape,
        "dynamic_shape": dynamic_shape,
        "row_sum":       row_sum,
        "softmax_first": softmax[0],
        "gathered":      gathered,
    }

mat = tf.random.normal([4, 6])
ops = graph_tensor_ops(mat)
for k, v in ops.items():
    print(f"  {k}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — CUSTOM GRADIENTS: OVERRIDING BACKPROP IN GRAPHS
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Numerically stable gradients (log-sum-exp, log1p),
#            straight-through estimator (binary neurons),
#            gradient clipping at individual op level,
#            non-differentiable operations (sorting, sampling).

@tf.custom_gradient
def stable_log(x):
    """
    Forward : log(x)  — same as tf.math.log
    Custom backward: clip gradients to [-1, 1] for stability.
    
    USE CASE: Numerical stability when x is near 0,
               preventing gradient explosion in log-probability computations.
    """
    y = tf.math.log(x + 1e-10)          # numerically stable forward

    def grad(dy):
        """dy = upstream gradient (∂loss/∂y). Returns ∂loss/∂x."""
        dx = dy / (x + 1e-10)
        return tf.clip_by_value(dx, -1.0, 1.0)  # clip to prevent explosion

    return y, grad


@tf.custom_gradient
def straight_through_round(x):
    """
    Straight-through estimator for rounding (used in VQ-VAE).
    Forward : round(x)   — non-differentiable
    Backward: treat as identity (pass gradients straight through)
    
    USE CASE: Vector quantization, binary networks, discrete latent variables.
    """
    def grad(dy):
        return dy   # gradient passes through as if round() didn't exist

    return tf.round(x), grad


@tf.function
def test_custom_gradients():
    x  = tf.constant([0.5, 1.0, 2.0])

    with tf.GradientTape() as tape:
        tape.watch(x)
        y1 = stable_log(x)
        y2 = straight_through_round(x)
        loss = tf.reduce_sum(y1 + y2)

    grads = tape.gradient(loss, x)
    tf.print("Custom gradient values:", grads)
    return grads

test_custom_gradients()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — SAVEDMODEL: EXPORTING & IMPORTING GRAPHS
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Deploying models to TF Serving, TFLite, TF.js,
#            sharing models across teams, language-agnostic serving (C++, Java).

class ProductionModel(tf.Module):
    """
    A complete model with multiple serving signatures.
    USE CASE: REST API with different input formats
               (raw features vs pre-processed vs batch vs single).
    """
    def __init__(self):
        super().__init__()
        self.W1 = tf.Variable(tf.random.normal([10, 32]), name="W1")
        self.b1 = tf.Variable(tf.zeros([32]), name="b1")
        self.W2 = tf.Variable(tf.random.normal([32, 5]), name="W2")
        self.b2 = tf.Variable(tf.zeros([5]), name="b2")

    def forward(self, x):
        h = tf.nn.relu(tf.linalg.matmul(x, self.W1) + self.b1)
        return tf.linalg.matmul(h, self.W2) + self.b2

    @tf.function(input_signature=[
        tf.TensorSpec([None, 10], tf.float32, name="features")])
    def serve_batch(self, features):
        """Signature 1: batch inference (for TF Serving REST API)."""
        logits = self.forward(features)
        return {"logits": logits,
                "probabilities": tf.nn.softmax(logits),
                "class": tf.argmax(logits, axis=-1)}

    @tf.function(input_signature=[
        tf.TensorSpec([10], tf.float32, name="single_feature")])
    def serve_single(self, feature):
        """Signature 2: single-sample inference (for edge devices)."""
        x      = tf.expand_dims(feature, 0)
        logits = self.forward(x)
        return {"class": tf.argmax(logits, axis=-1)[0],
                "confidence": tf.reduce_max(tf.nn.softmax(logits))}

    @tf.function(input_signature=[
        tf.TensorSpec([None, 10], tf.float32, name="raw_input"),
        tf.TensorSpec([10], tf.float32, name="mean"),
        tf.TensorSpec([10], tf.float32, name="std")])
    def serve_with_preprocessing(self, raw_input, mean, std):
        """Signature 3: includes normalization in graph (zero external deps)."""
        normalized = (raw_input - mean) / (std + 1e-8)
        return self.serve_batch(normalized)


prod_model = ProductionModel()

# Save with multiple signatures
tf.saved_model.save(
    prod_model,
    "/tmp/production_model",
    signatures={
        "serving_default":      prod_model.serve_batch,
        "single":               prod_model.serve_single,
        "with_preprocessing":   prod_model.serve_with_preprocessing,
    }
)
print("\nSavedModel exported to /tmp/production_model")

# Load and inspect
loaded = tf.saved_model.load("/tmp/production_model")
print(f"Available signatures: {list(loaded.signatures.keys())}")

# Run via signature (as TF Serving would)
batch_fn  = loaded.signatures["serving_default"]
test_inp  = tf.random.normal([3, 10])
output    = batch_fn(features=test_inp)
print(f"Serving output keys   : {list(output.keys())}")
print(f"Predicted classes     : {output['class'].numpy()}")

# Inspect saved graph (useful for debugging deployments)
print("\n─── SavedModel MetaGraph tags ───")
meta = tf.saved_model.load("/tmp/production_model")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — GRAPH TRACING PITFALLS & BEST PRACTICES
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Avoiding common bugs that cause silent correctness issues
#            or excessive retracing (performance degradation).

print("\n─── PITFALL 1: Python side-effects in @tf.function ───")

side_effects = []

@tf.function
def fn_with_side_effect(x):
    """
    PROBLEM: List append runs only during TRACE, not every call.
    SOLUTION: Use tf.TensorArray or log outside @tf.function.
    """
    side_effects.append(x)    # ← WRONG: only runs during trace
    tf.print("TF print works:", x)  # ← CORRECT: runs every call
    return x * 2

fn_with_side_effect(tf.constant(1.0))
fn_with_side_effect(tf.constant(2.0))
print(f"Python list length: {len(side_effects)} (expected 2, got 1 — bug!)")


print("\n─── PITFALL 2: Python integers cause retracing ───")

@tf.function
def fn_with_python_int(x, n):
    """
    PROBLEM: Python int n creates a NEW trace per unique value.
    SOLUTION: Cast n to tf.Tensor before passing, or use tf.constant.
    """
    return x * n

fn_with_python_int(tf.constant(1.0), 2)   # trace 1
fn_with_python_int(tf.constant(1.0), 3)   # trace 2 ← retrace!
fn_with_python_int(tf.constant(1.0), tf.constant(4))  # trace 3 ← retrace!

@tf.function
def fn_with_tensor_int(x, n):
    """FIXED: n is always a tf.Tensor — single trace."""
    return x * tf.cast(n, x.dtype)

fn_with_tensor_int(tf.constant(1.0), tf.constant(2))
fn_with_tensor_int(tf.constant(1.0), tf.constant(3))   # reuses same trace ✓


print("\n─── PITFALL 3: Variable creation inside @tf.function ───")

@tf.function
def bad_variable_creation(x):
    """
    PROBLEM: tf.Variable created inside @tf.function on first trace.
    On second trace with different shape → ERROR (can't resize variable).
    SOLUTION: Create variables at module/class level, pass as arguments.
    """
    # v = tf.Variable(tf.zeros_like(x))  # ← DANGEROUS
    # CORRECT: receive pre-created variable as argument:
    return x + 1   # placeholder

print("Variable creation pitfall demonstrated (creation moved outside)")


print("\n─── BEST PRACTICE: Using tf.debugging inside graphs ───")

@tf.function
def safe_divide(a, b):
    """
    tf.debugging assertions become graph ops.
    USE CASE: Input validation in production — checks run at graph speed.
    Set tf.debugging.disable_check_numerics() to disable in deployment.
    """
    tf.debugging.assert_greater(b, tf.zeros_like(b),
                                  message="Denominator must be > 0")
    tf.debugging.check_numerics(a, message="Input a has NaN/Inf")
    return a / b

try:
    result = safe_divide(tf.constant([1.0, 2.0]),
                          tf.constant([2.0, 4.0]))
    tf.print("Safe divide:", result)
except tf.errors.InvalidArgumentError as e:
    print(f"Caught expected error: {e.message}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — GRAPH PROFILING WITH tf.profiler
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: Identifying bottlenecks (memory-bound vs compute-bound ops),
#            verifying GPU utilization, finding redundant kernel launches.

@tf.function
def profile_target(x):
    """Workload to profile."""
    for _ in range(20):
        x = tf.linalg.matmul(x, tf.transpose(x))
        x = tf.nn.relu(x)
    return x

mat = tf.random.normal([128, 128])

# Run profiler
tf.profiler.experimental.start("/tmp/profile_logs")
for _ in range(5):
    profile_target(mat)
tf.profiler.experimental.stop()
print("\nProfile saved to /tmp/profile_logs")
print("View with: tensorboard --logdir /tmp/profile_logs")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — FULL GRAPH TRAINING PIPELINE (putting it all together)
# ══════════════════════════════════════════════════════════════════════════════
# USE CASE: End-to-end production training loop with all graph features:
#            custom model, custom loss, GradientTape, metrics, callbacks.

class GraphModel(tf.Module):
    """Simple 2-layer model using raw tf.Module + @tf.function."""

    def __init__(self, in_dim=20, hidden=64, out_dim=5, name="GraphModel"):
        super().__init__(name=name)
        # All weights created at __init__ time (outside any @tf.function)
        self.W1  = tf.Variable(tf.initializers.GlorotUniform()([in_dim, hidden]),
                                name="W1")
        self.b1  = tf.Variable(tf.zeros([hidden]), name="b1")
        self.W2  = tf.Variable(tf.initializers.GlorotUniform()([hidden, out_dim]),
                                name="W2")
        self.b2  = tf.Variable(tf.zeros([out_dim]), name="b2")
        self.bn_gamma = tf.Variable(tf.ones([hidden]),  name="bn_gamma")
        self.bn_beta  = tf.Variable(tf.zeros([hidden]), name="bn_beta")

    @tf.function(input_signature=[
        tf.TensorSpec([None, 20], tf.float32),
        tf.TensorSpec([], tf.bool)])
    def __call__(self, x, training):
        h = tf.linalg.matmul(x, self.W1) + self.b1

        # Manual batch normalization
        if training:
            mean, var = tf.nn.moments(h, axes=[0])
        else:
            mean = tf.reduce_mean(h, axis=0)
            var  = tf.math.reduce_variance(h, axis=0)

        h    = tf.nn.batch_normalization(
            h, mean, var, self.bn_beta, self.bn_gamma, 1e-6)
        h    = tf.nn.relu(h)
        return tf.linalg.matmul(h, self.W2) + self.b2


@tf.function
def compute_metrics(logits, labels):
    """Graph-compatible accuracy computation."""
    preds   = tf.argmax(logits, axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(preds, labels), tf.float32)
    return tf.reduce_mean(correct)


def full_graph_training(epochs=5, batch_size=64, lr=1e-3):
    # ── Data ─────────────────────────────────────────────────────────────────
    N = 2000
    X = tf.constant(np.random.randn(N, 20).astype(np.float32))
    y = tf.constant(np.random.randint(0, 5, N).astype(np.int32))

    train_ds = (tf.data.Dataset.from_tensor_slices((X, y))
                .shuffle(1000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    # ── Model & optimizer ────────────────────────────────────────────────────
    model     = GraphModel()
    optimizer = tf.keras.optimizers.AdamW(lr, weight_decay=1e-4)

    # ── Metrics (graph-compatible) ───────────────────────────────────────────
    loss_avg   = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Mean()

    # ── Single compiled train step ────────────────────────────────────────────
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=tf.constant(True))
            y_oh   = tf.one_hot(y_batch, depth=5)
            loss   = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(y_oh, logits))

        grads = tape.gradient(loss, model.trainable_variables)
        # Gradient clipping inside graph
        grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = compute_metrics(logits, y_batch)
        return loss, acc

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        loss_avg.reset_state()
        acc_metric.reset_state()

        for x_b, y_b in train_ds:
            loss, acc = train_step(x_b, y_b)
            loss_avg.update_state(loss)
            acc_metric.update_state(acc)

        print(f"Epoch {epoch}/{epochs}  "
              f"loss={loss_avg.result():.4f}  "
              f"acc={acc_metric.result():.4f}")

    # ── Export ────────────────────────────────────────────────────────────────
    tf.saved_model.save(model, "/tmp/graph_model_final",
                          signatures={"serving_default": model.__call__})
    print("\nFinal model saved to /tmp/graph_model_final")
    return model


print("\n" + "="*60)
print("  FULL GRAPH TRAINING PIPELINE")
print("="*60)
trained_model = full_graph_training(epochs=3)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY & QUICK-REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
"""
QUICK REFERENCE — WHEN TO USE WHAT
──────────────────────────────────────────────────────────────────────────────
@tf.function              → compile any Python function to a graph
input_signature=          → freeze shapes; prevents retracing; required for export
concrete_function         → single compiled graph; lowest dispatch overhead
tf.cond / tf.switch_case  → runtime branching inside graphs
tf.while_loop             → dynamic iteration (unknown steps at trace time)
tf.TensorArray            → dynamic accumulation inside graphs
tf.custom_gradient        → override backpropagation (STE, stability)
GradientTape              → explicit differentiation; required for custom loops
tf.Variable               → stateful, persistent; always create outside @tf.function
tf.Module                 → portable, composable, auto-tracks variables
tf.saved_model.save/load  → serialize full graph + weights for serving
tf.profiler               → find performance bottlenecks
tf.debugging              → graph-speed assertions for production safety

GOLDEN RULES
──────────────────────────────────────────────────────────────────────────────
1. Create tf.Variable OUTSIDE @tf.function
2. Avoid Python side-effects (list.append, print) inside @tf.function
3. Pass dynamic values as tf.Tensor, not Python scalars → prevents retracing
4. Use tf.print (not print) for runtime logging in graphs
5. Use input_signature when the function will be exported/served
6. Profile before optimizing — measure, don't guess
"""

print("\n✅ All graph sections executed successfully.")
print("   Open TensorBoard for profiling: tensorboard --logdir /tmp")