# PyPTO 处理流程详解

## 概述

本文档以 `examples/language/beginner/hello_world.py` (简单的张量加法) 为例，拆解 PyPTO 的完整处理流程，从 Python DSL 到最终生成的代码。

## 用例示例：Hello World (张量加法)

### 输入代码

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

## 阶段 1: 用户 API 调用 (Python DSL)

### 1.1 入口：装饰器语法

| 装饰器 | 作用 |
|--------|------|
| `@pl.program` | 标记一个类为 PyPTO 程序 |
| `@pl.function` | 标记一个方法为 PyPTO 函数 |

### 1.2 函数类型

| 类型 | 说明 |
|------|------|
| `InCore` | 在 AI 核心上执行的计算函数，使用 block-level 操作 |
| `Orchestration` | 调度函数，负责调用 InCore 函数并协调执行 |

### 1.3 操作语义

DSL 操作如 `pl.load()` 是语法糖，在 IR 构建过程中会转换为 block-level 操作：

```python
# DSL 源代码（用户编写）:
tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
# 内部转换为前端 IR:
# tile_a = pl.block.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Vec)
```

---

## 运行示例

要查看完整的处理流程，运行以下命令：

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

这会在 `build_output/HelloWorldProgram_<timestamp>/` 目录下生成所有中间输出以供检查。

---

## 阶段 2: IR 构建 (Intermediate Representation)

### 2.1 IR 层次结构

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

### 2.2 核心 IR 节点

| 节点类型 | 文件位置 | 用途 |
|----------|----------|------|
| `Program` | `include/pypto/ir/program.h` | 顶层容器 |
| `Function` | `include/pypto/ir/function.h` | 函数定义 |
| `Var` | `include/pypto/ir/expr.h` | 变量引用 |
| `Call` | `include/pypto/ir/expr.h` | 操作调用 |
| `AssignStmt` | `include/pypto/ir/stmt.h` | 赋值语句 |
| `ReturnStmt` | `include/pypto/ir/stmt.h` | 返回语句 |

### 2.3 类型系统

| 类型 | 说明 |
|------|------|
| `TensorType` | 张量类型，包含形状和数据类型 |
| `TileType` | Tile 类型，硬件感知的数据块 |
| `ScalarType` | 标量类型 |
| `MemRef` | 内存引用，描述数据在内存中的布局 |

---

## 阶段 3: Pass 变换 (IR Transforms)

### 3.1 Pass 执行流程

```
输入 IR (Frontend)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: ConvertToSSA                                        │
│ - 将变量转换为 SSA 形式 (静态单赋值)                         │
│ - 确保每个变量只被赋值一次                                   │
│ 产出: IRProperty::SSAForm                                   │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 2: FlattenCallExpr                                     │
│ - 展平嵌套的函数调用表达式                                   │
│ - 例如: add(load(a), load(b)) → t1=load(a); t2=load(b); add │
│ 产出: IRProperty::NoNestedCalls                             │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 3: RunVerifier                                         │
│ - 验证 IR 结构和类型正确性                                   │
│ 产出: IRProperty::TypeChecked                               │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 4: InitMemRef                                          │
│ - 初始化内存引用 (MemRef)                                    │
│ - 为每个 Tile 变量分配内存描述符                             │
│ 产出: IRProperty::HasMemRefs                                │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 5: MemoryReuse                                         │
│ - 分析 Tile 生命周期                                         │
│ - 复用不再使用的 Tile 内存                                   │
│ - 减少总内存占用                                             │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 6: InsertSync (仅 Default 策略)                        │
│ - 插入同步操作 (system.msync 等)                            │
│ - 确保多核执行时的数据一致性                                 │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 7: AllocateMemoryAddr                                  │
│ - 分配具体的内存地址                                         │
│ - 确定每个 MemRef 的起始地址和大小                           │
│ 产出: IRProperty::AllocatedMemoryAddr                       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
输出 IR (Transformed)
```

### 3.2 优化策略对比

| Pass | Default | PTOAS |
|------|---------|-------|
| ConvertToSSA | ✓ | ✓ |
| FlattenCallExpr | ✓ | ✓ |
| RunVerifier | ✓ | ✓ |
| InitMemRef | ✓ | ✓ |
| MemoryReuse | ✓ | ✓ |
| InsertSync | ✓ | ✗ |
| AllocateMemoryAddr | ✓ | ✓ |

**区别**: `PTOAS` 策略跳过 `InsertSync`，用于生成纯 PTO 汇编。

### 3.3 关键文件

| 文件 | 说明 |
|------|------|
| `python/pypto/ir/pass_manager.py` | Pass 管理器，定义策略 |
| `src/ir/transforms/passes.cpp` | Pass 注册和工厂函数 |
| `src/ir/transforms/convert_to_ssa_pass.cpp` | SSA 转换实现 |
| `src/ir/transforms/init_memref.cpp` | MemRef 初始化 |
| `src/ir/transforms/basic_memory_reuse_pass.cpp` | 内存复用优化 |
| `src/ir/transforms/insert_sync_pass.cpp` | 同步插入 |

### 3.4 IR 属性

Pass 产出并验证 IR 属性以确保正确性：

| 属性 | 描述 | 产出者 |
|------|------|--------|
| `SSAForm` | IR 为 SSA 形式（变量只赋值一次） | ConvertToSSA |
| `TypeChecked` | 类型检查通过 | RunVerifier |
| `NoNestedCalls` | 无嵌套调用表达式 | FlattenCallExpr |
| `HasMemRefs` | MemRefs 已初始化 | InitMemRef |
| `AllocatedMemoryAddr` | 内存地址已分配 | AllocateMemoryAddr |

### 3.5 验证系统

验证系统通过 `PassContext` 控制：

| 验证级别 | 行为 |
|----------|------|
| `None` | 不自动验证 |
| `Basic` | 验证轻量级属性一次（默认） |

通过 `PYPTO_VERIFY_LEVEL` 环境变量或 `PassContext` 构造函数设置。

---

## 阶段 4: CodeGen 代码生成

### 4.1 后端选择

| 后端 | 输出格式 | 文件位置 |
|------|----------|----------|
| **PTO** | PTO-ISA MLIR | `src/codegen/pto/` |
| **CCE** | C++ Kernel | `src/codegen/cce/` |

### 4.1.1 skip_ptoas 参数

| 值 | 输出 | kernel_config.py |
|----|------|------------------|
| `True` | 原始 MLIR (.pto) 文件 | 不生成 |
| `False` | 通过 ptoas 工具生成的 C++ 包装器 (.cpp) | 生成 |

### 4.2 PTO CodeGen 流程

```
PTOCodegen.Generate(Program)
  │
  ├── 遍历所有 Function
  │     │
  │     ├── 收集 MemRef 信息
  │     │
  │     ├── 生成函数签名
  │     │   func.func @tile_add(%arg0: !pto.ptr<f32>, ...)
  │     │
  │     ├── 生成 TensorView
  │     │   %0 = pto.make_tensor_view %arg0, shape=[128, 128]
  │     │
  │     ├── 分配 Tile 缓冲区
  │     │   %tile_a = pto.alloc_tile : !pto.tile_buf<...>
  │     │
  │     └── 遍历函数体语句
  │           ├── AssignStmt → 发射操作
  │           │     block.load → pto.tload
  │           │     block.add → pto.tadd
  │           │     block.store → pto.tstore
  │           │
  │           └── ReturnStmt → func.return
  │
  └── 输出 MLIR 模块
```

### 4.3 操作映射表

| PyPTO 操作 | PTO-ISA MLIR 操作 |
|------------|-------------------|
| `block.load` | `pto.partition_view` + `pto.tload` |
| `block.store` | `pto.partition_view` + `pto.tstore` |
| `block.add` | `pto.tadd` |
| `block.mul` | `pto.tmul` |
| `block.matmul` | `pto.tmm` |

### 4.4 生成的 MLIR 示例

实际生成的 `tile_add` MLIR:

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
    pto.tload ins(%6 : !pto.partition_tensor_view<128x128xf32>)
              outs(%0 : !pto.tile_buf<...>)
    %7 = pto.partition_view %4, offsets = [%c0, %c0], sizes = [%c128, %c128]
             : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<128x128xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<128x128xf32>)
              outs(%1 : !pto.tile_buf<...>)
    pto.tadd ins(%0, %1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%2 : !pto.tile_buf<...>)
    %8 = pto.partition_view %5, offsets = [%c0, %c0], sizes = [%c128, %c128]
             : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<128x128xf32>
    pto.tstore ins(%2 : !pto.tile_buf<...>)
               outs(%8 : !pto.partition_tensor_view<128x128xf32>)
    return
  }
}
```

### 4.5 Orchestration 代码生成

对于 `Orchestration` 函数，PyPTO 生成 C++ 编排代码

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

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户代码 (Python DSL)                              │
│  examples/language/beginner/hello_world.py                                  │
│                                                                             │
│  @pl.program                                                                │
│  class HelloWorldProgram:                                                   │
│      @pl.function(type=pl.FunctionType.InCore)                             │
│      def tile_add(self, a, b, c): ...                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IR 构建 (IRBuilder)                                │
│  python/pypto/language/ → pypto_core (C++ bindings)                         │
│                                                                             │
│  Program                                                                    │
│  ├── Function (orchestrator)                                                │
│  │   └── Body: [AssignStmt, Call, ReturnStmt]                              │
│  └── Function (tile_add)                                                    │
│      └── Body: [AssignStmt(block.load), AssignStmt(block.add), ...]        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pass 变换 (PassPipeline)                           │
│  python/pypto/ir/pass_manager.py → src/ir/transforms/                       │
│                                                                             │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │ ConvertToSSA     │ → │ FlattenCallExpr  │ → │ RunVerifier      │        │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│           │                                              │                  │
│           ▼                                              ▼                  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │ InitMemRef       │ → │ MemoryReuse      │ → │ InsertSync       │        │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ AllocateMemoryAddr│                                                      │
│  └──────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CodeGen (PTO Backend)                              │
│  src/codegen/pto/pto_codegen.cpp                                            │
│                                                                             │
│  IR → MLIR (PTO-ISA)                                                        │
│  ├── pto.make_tensor_view                                                   │
│  ├── pto.alloc_tile                                                         │
│  ├── pto.tload / pto.tstore                                                 │
│  └── pto.tadd / pto.tmul / pto.tmm                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输出文件                                           │
│  build_output/HelloWorldProgram_<timestamp>/                                │
│                                                                             │
│  ├── passes_dump/                    # Pass 中间结果                         │
│  │   ├── 00_frontend.py                                                     │
│  │   ├── 01_after_ConvertToSSA.py                                           │
│  │   ├── 02_after_FlattenCallExpr.py                                        │
│  │   ├── 03_after_RunVerifier.py                                            │
│  │   ├── 04_after_InitMemRef.py                                             │
│  │   ├── 05_after_MemoryReuse.py                                            │
│  │   └── 06_after_AllocateMemoryAddr.py                                     │
│  ├── report/                         # 内存报告                              │
│  │   └── memory_after_AllocateMemoryAddr.txt                                │
│  ├── kernels/                        # 生成的 MLIR                           │
│  │   └── aiv/                        # AI Vector kernels                     │
│  │       └── tile_add.pto            # (skip_ptoas=True 时为 .pto)           │
│  └── orchestration/                  # 编排代码                              │
│      └── orchestrator.cpp                                                   │
│                                                                             │
│  注：kernel_config.py 仅在 skip_ptoas=False 时生成                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 关键文件索引

### 用户 API 层
- `examples/language/beginner/hello_world.py` - 入门示例
- `examples/ir_builder/flash_attention_builder.py` - 复杂示例
- `python/pypto/language/` - DSL 实现

### IR 核心层
- `include/pypto/ir/core.h` - IRNode 基类
- `include/pypto/ir/expr.h` - 表达式节点
- `include/pypto/ir/stmt.h` - 语句节点
- `include/pypto/ir/type.h` - 类型系统
- `include/pypto/ir/builder.h` - IRBuilder API

### Pass 系统层
- `python/pypto/ir/pass_manager.py` - Pass 管理器
- `python/pypto/ir/compile.py` - 编译入口
- `src/ir/transforms/passes.cpp` - Pass 注册
- `src/ir/transforms/*.cpp` - 各 Pass 实现

### CodeGen 层
- `include/pypto/codegen/pto/pto_codegen.h` - PTO CodeGen 头文件
- `src/codegen/pto/pto_codegen.cpp` - PTO CodeGen 实现
- `src/codegen/cce/cce_codegen.cpp` - CCE CodeGen 实现

### 相关文档
- [IR 概述](ir/00-overview.md) - IR 基础
- [Pass 系统](passes/00-pass_manager.md) - Pass 系统文档
- [PTO CodeGen](codegen/00-pto_codegen.md) - CodeGen 文档
- [Python 语法](language/00-python_syntax.md) - DSL 参考
