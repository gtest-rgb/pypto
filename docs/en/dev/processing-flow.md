# PyPTO Processing Flow

## Overview

This document uses `examples/language/beginner/hello_world.py` (simple tensor addition) to explain PyPTO's complete processing pipeline from Python DSL to generated code.

## Example: Hello World (Tensor Addition)

### Input Code

```python
import pypto.language as pl

@pl.program
class HelloWorldProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
        tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
        tile_c = pl.add(tile_a, tile_b)
        out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
        return out_c

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        out_c = self.tile_add(a, b, out_c)
        return out_c
```

---

## Stage 1: User API (Python DSL)

### 1.1 Decorator Syntax

| Decorator | Purpose |
|-----------|---------|
| `@pl.program` | Marks a class as a PyPTO program |
| `@pl.function` | Marks a method as a PyPTO function |

### 1.2 Function Types

| Type | Description |
|------|-------------|
| `InCore` | Compute function executing on AI core, uses block-level operations |
| `Orchestration` | Scheduling function that calls InCore functions and coordinates execution |

### 1.3 Operation Semantics

DSL operations like `pl.load()` are syntactic sugar transformed to block-level operations:

```python
# DSL source (user writes):
tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
# Transformed internally to:
# tile_a = pl.block.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec)
```

---

## Running the Example

To see the complete processing flow in action, run:

```bash
python -c "
from examples.language.beginner.hello_world import HelloWorldProgram
from pypto import ir

output_dir = ir.compile(
    HelloWorldProgram,
    strategy=ir.OptimizationStrategy.PTOAS,
    skip_ptoas=True,
    dump_passes=True,
)
print(f'Output directory: {output_dir}')
"
```

This generates all intermediate outputs for inspection in `build_output/HelloWorldProgram_<timestamp>/`.

---

## Stage 2: IR Construction

### 2.1 IR Hierarchy

```
Program
└── Function (orchestrator)
    ├── Params: a, b
    ├── ReturnTypes: Tensor[[128,128], FP32]
    └── Body: ScopeStmt
        ├── AssignStmt: out_c = create_tensor(...)
        ├── AssignStmt: out_c = call tile_add(a, b, out_c)
        └── ReturnStmt: return out_c

└── Function (tile_add)
    ├── Params: a, b, c
    ├── ReturnTypes: Tensor[[128,128], FP32]
    └── Body: ScopeStmt
        ├── AssignStmt: tile_a = block.load(a, ...)
        ├── AssignStmt: tile_b = block.load(b, ...)
        ├── AssignStmt: tile_c = block.add(tile_a, tile_b)
        ├── AssignStmt: out_c = block.store(tile_c, ...)
        └── ReturnStmt: return out_c
```

### 2.2 Core IR Nodes

| Node Type | File | Purpose |
|-----------|------|---------|
| `Program` | `include/pypto/ir/program.h` | Top-level container |
| `Function` | `include/pypto/ir/function.h` | Function definition |
| `Var` | `include/pypto/ir/expr.h` | Variable reference |
| `Call` | `include/pypto/ir/expr.h` | Operation call |
| `AssignStmt` | `include/pypto/ir/stmt.h` | Assignment statement |
| `ReturnStmt` | `include/pypto/ir/stmt.h` | Return statement |

### 2.3 Type System

| Type | Description |
|------|-------------|
| `TensorType` | Tensor type with shape and data type |
| `TileType` | Tile type, hardware-aware data block |
| `ScalarType` | Scalar type |
| `MemRef` | Memory reference describing data layout in memory |

---

## Stage 3: Pass Transforms

### 3.1 Pass Execution Flow

```
Input IR (Frontend)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: ConvertToSSA                                        │
│ - Convert variables to SSA form (Static Single Assignment)  │
│ - Ensure each variable is assigned only once                │
│ Produces: IRProperty::SSAForm                               │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 2: FlattenCallExpr                                     │
│ - Flatten nested function call expressions                  │
│ - e.g., add(load(a), load(b)) → t1=load(a); t2=load(b); add │
│ Produces: IRProperty::NoNestedCalls                         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 3: RunVerifier                                         │
│ - Verify IR structure and type correctness                  │
│ Produces: IRProperty::TypeChecked                           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 4: InitMemRef                                          │
│ - Initialize memory references (MemRef)                     │
│ - Allocate memory descriptors for each Tile variable        │
│ Produces: IRProperty::HasMemRefs                            │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 5: MemoryReuse                                         │
│ - Analyze Tile lifetimes                                    │
│ - Reuse memory from Tiles no longer in use                  │
│ - Reduce total memory footprint                             │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 6: InsertSync (Default strategy only)                  │
│ - Insert synchronization operations (system.msync, etc.)    │
│ - Ensure data consistency during multi-core execution       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 7: AllocateMemoryAddr                                  │
│ - Allocate specific memory addresses                        │
│ - Determine start address and size for each MemRef          │
│ Produces: IRProperty::AllocatedMemoryAddr                   │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
Output IR (Transformed)
```

### 3.2 Strategy Comparison

| Pass | Default | PTOAS |
|------|---------|-------|
| ConvertToSSA | ✓ | ✓ |
| FlattenCallExpr | ✓ | ✓ |
| RunVerifier | ✓ | ✓ |
| InitMemRef | ✓ | ✓ |
| MemoryReuse | ✓ | ✓ |
| InsertSync | ✓ | ✗ |
| AllocateMemoryAddr | ✓ | ✓ |

**Difference**: `PTOAS` strategy skips `InsertSync`, used for generating pure PTO assembly.

### 3.3 Key Files

| File | Description |
|------|-------------|
| `python/pypto/ir/pass_manager.py` | Pass manager, defines strategies |
| `src/ir/transforms/passes.cpp` | Pass registration and factory functions |
| `src/ir/transforms/convert_to_ssa_pass.cpp` | SSA conversion implementation |
| `src/ir/transforms/init_memref.cpp` | MemRef initialization |
| `src/ir/transforms/basic_memory_reuse_pass.cpp` | Memory reuse optimization |
| `src/ir/transforms/insert_sync_pass.cpp` | Synchronization insertion |

### 3.4 IR Properties

Passes produce and verify IR properties to ensure correctness:

| Property | Description | Produced By |
|----------|-------------|-------------|
| `SSAForm` | IR in SSA form (variables assigned once) | ConvertToSSA |
| `TypeChecked` | Type checking passed | Preserved by ConvertToSSA, FlattenCallExpr |
| `NoNestedCalls` | No nested call expressions | FlattenCallExpr |
| `HasMemRefs` | MemRefs initialized | InitMemRef |
| `AllocatedMemoryAddr` | Memory addresses allocated | AllocateMemoryAddr |

### 3.5 Verification System

The verification system is controlled via `PassContext`:

| Verification Level | Behavior |
|-------------------|----------|
| `None` | No automatic verification |
| `Basic` | Verify lightweight properties once (default) |

Set via `PYPTO_VERIFY_LEVEL` environment variable or `PassContext` constructor.

---

## Stage 4: Code Generation

### 4.1 Backend Selection

| Backend | Output Format | File Location |
|---------|---------------|---------------|
| **PTO** | PTO-ISA MLIR | `src/codegen/pto/` |
| **CCE** | C++ Kernel | `src/codegen/cce/` |

### 4.1.1 skip_ptoas Parameter

| Value | Output | kernel_config.py |
|-------|--------|------------------|
| `True` | Raw MLIR (.pto) files | Not generated |
| `False` | C++ wrappers (.cpp) via ptoas tool | Generated |

### 4.2 PTO CodeGen Flow

```
PTOCodegen.Generate(Program)
  │
  ├── Iterate all Functions
  │     │
  │     ├── Collect MemRef information
  │     │
  │     ├── Generate function signature
  │     │   func.func @tile_add(%arg0: !pto.ptr<f32>, ...)
  │     │
  │     ├── Generate TensorView
  │     │   %0 = pto.make_tensor_view %arg0, shape=[128, 128]
  │     │
  │     ├── Allocate Tile buffers
  │     │   %tile_a = pto.alloc_tile : !pto.tile_buf<...>
  │     │
  │     └── Iterate function body statements
  │           ├── AssignStmt → Emit operations
  │           │     block.load → pto.tload
  │           │     block.add → pto.tadd
  │           │     block.store → pto.tstore
  │           │
  │           └── ReturnStmt → func.return
  │
  └── Output MLIR module
```

### 4.3 Operation Mapping

| PyPTO Operation | PTO-ISA MLIR Operation |
|-----------------|------------------------|
| `block.load` | `pto.partition_view` + `pto.tload` |
| `block.store` | `pto.partition_view` + `pto.tstore` |
| `block.add` | `pto.tadd` |
| `block.mul` | `pto.tmul` |
| `block.matmul` | `pto.tmm` |

### 4.4 Generated MLIR Example

The actual generated MLIR for `tile_add`:

```mlir
module {
  func.func @tile_add(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>, %arg2: !pto.ptr<f32>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %3 = pto.make_tensor_view %arg0, shape = [%c128, %c128] strides = [%c128, %c1]
             : !pto.tensor_view<?x?xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c128, %c128] strides = [%c128, %c1]
             : !pto.tensor_view<?x?xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c128, %c128] strides = [%c128, %c1]
             : !pto.tensor_view<?x?xf32>
    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=128, cols=128,
             v_row=128, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=128, cols=128,
             v_row=128, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=128, cols=128,
             v_row=128, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %6 = pto.partition_view %3, offsets = [%c0, %c0], sizes = [%c128, %c128]
             : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<128x128xf32>
    pto.tload ins(%6 : !pto.partition_tensor_view<128x128xf32>) outs(%0 : !pto.tile_buf<...>)
    %7 = pto.partition_view %4, offsets = [%c0, %c0], sizes = [%c128, %c128]
             : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<128x128xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<128x128xf32>) outs(%1 : !pto.tile_buf<...>)
    pto.tadd ins(%0, %1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%2 : !pto.tile_buf<...>)
    %8 = pto.partition_view %5, offsets = [%c0, %c0], sizes = [%c128, %c128]
             : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<128x128xf32>
    pto.tstore ins(%2 : !pto.tile_buf<...>) outs(%8 : !pto.partition_tensor_view<128x128xf32>)
    return
  }
}
```

### 4.5 Orchestration Code Generation

For `Orchestration` functions, PyPTO generates C++ orchestration code:

```cpp
// Orchestration Function: orchestrator
extern "C" {
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    void* arg_a_ptr = reinterpret_cast<void*>(args[0]);
    void* arg_b_ptr = reinterpret_cast<void*>(args[1]);
    void* arg_out_c_ptr = reinterpret_cast<void*>(args[2]);

    // Create external tensors
    Tensor ext_a = make_tensor_external(arg_a_ptr, {128, 128}, 2, DataType::FLOAT32);
    Tensor ext_b = make_tensor_external(arg_b_ptr, {128, 128}, 2, DataType::FLOAT32);
    Tensor ext_out_c = make_tensor_external(arg_out_c_ptr, {128, 128}, 2, DataType::FLOAT32);

    // Submit tile_add task
    PTOParam params[] = {
        make_input_param(ext_a),
        make_input_param(ext_b),
        make_output_param(ext_out_c),
    };
    pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, params, 3);
}
}
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Code (Python DSL)                             │
│  examples/language/beginner/hello_world.py                                   │
│                                                                              │
│  @pl.program                                                                 │
│  class HelloWorldProgram:                                                    │
│      @pl.function(type=pl.FunctionType.InCore)                              │
│      def tile_add(self, a, b, c): ...                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IR Construction (IRBuilder)                        │
│  python/pypto/language/ → pypto_core (C++ bindings)                          │
│                                                                              │
│  Program                                                                     │
│  ├── Function (orchestrator)                                                 │
│  │   └── Body: [AssignStmt, Call, ReturnStmt]                               │
│  └── Function (tile_add)                                                     │
│      └── Body: [AssignStmt(block.load), AssignStmt(block.add), ...]         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pass Transforms (PassPipeline)                     │
│  python/pypto/ir/pass_manager.py → src/ir/transforms/                        │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐         │
│  │ ConvertToSSA     │ → │ FlattenCallExpr  │ → │ RunVerifier      │         │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘         │
│           │                                               │                   │
│           ▼                                               ▼                   │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐         │
│  │ InitMemRef       │ → │ MemoryReuse      │ → │ InsertSync       │         │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ AllocateMemoryAddr│                                                       │
│  └──────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CodeGen (PTO Backend)                              │
│  src/codegen/pto/pto_codegen.cpp                                             │
│                                                                              │
│  IR → MLIR (PTO-ISA)                                                         │
│  ├── pto.make_tensor_view                                                    │
│  ├── pto.alloc_tile                                                          │
│  ├── pto.tload / pto.tstore                                                  │
│  └── pto.tadd / pto.tmul / pto.tmm                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Output Files                                       │
│  build_output/HelloWorldProgram_<timestamp>/                                 │
│                                                                              │
│  ├── passes_dump/                    # Pass intermediate results             │
│  │   ├── 00_frontend.py                                                      │
│  │   ├── 01_after_ConvertToSSA.py                                            │
│  │   ├── 02_after_FlattenCallExpr.py                                         │
│  │   ├── 03_after_RunVerifier.py                                             │
│  │   ├── 04_after_InitMemRef.py                                              │
│  │   ├── 05_after_MemoryReuse.py                                             │
│  │   └── 06_after_AllocateMemoryAddr.py                                      │
│  ├── report/                         # Memory report                         │
│  │   └── memory_after_AllocateMemoryAddr.txt                                 │
│  ├── kernels/                        # Generated MLIR                        │
│  │   └── aiv/                        # AI Vector kernels                     │
│  │       └── tile_add.pto            # (.pto with skip_ptoas=True)           │
│  └── orchestration/                  # Orchestration code                     │
│      └── orchestrator.cpp                                                    │
│                                                                              │
│  Note: kernel_config.py is generated only when skip_ptoas=False              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key File Index

### User API Layer
- `examples/language/beginner/hello_world.py` - Getting started example
- `examples/ir_builder/flash_attention_builder.py` - Complex example
- `python/pypto/language/` - DSL implementation

### IR Core Layer
- `include/pypto/ir/core.h` - IRNode base class
- `include/pypto/ir/expr.h` - Expression nodes
- `include/pypto/ir/stmt.h` - Statement nodes
- `include/pypto/ir/type.h` - Type system
- `include/pypto/ir/builder.h` - IRBuilder API

### Pass System Layer
- `python/pypto/ir/pass_manager.py` - Pass manager
- `python/pypto/ir/compile.py` - Compile entry point
- `src/ir/transforms/passes.cpp` - Pass registration
- `src/ir/transforms/*.cpp` - Individual pass implementations

### CodeGen Layer
- `include/pypto/codegen/pto/pto_codegen.h` - PTO CodeGen header
- `src/codegen/pto/pto_codegen.cpp` - PTO CodeGen implementation
- `src/codegen/cce/cce_codegen.cpp` - CCE CodeGen implementation

### Related Documentation
- [IR Overview](ir/00-overview.md) - IR fundamentals
- [Pass Manager](passes/00-pass_manager.md) - Pass system documentation
- [PTO CodeGen](codegen/00-pto_codegen.md) - CodeGen documentation
- [Python Syntax](language/00-python_syntax.md) - DSL reference
