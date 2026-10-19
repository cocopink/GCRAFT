"""
Constraint State Management Module
用于管理和维护推理过程中的约束状态
"""

from typing import Set, List, Dict, Tuple, Any
import copy


class ConstraintState:
    """
    约束状态类，用于管理和维护推理过程中的约束
    
    核心功能：
    1. 存储各类约束（行、列、操作、聚合）
    2. 提供约束查询接口
    3. 维护约束历史记录
    4. 支持约束的增量更新
    """
    
    def __init__(self):
        # 行约束：禁止选择的行集合
        self.forbidden_rows: Set[str] = set()
        
        # 列约束：禁止选择的列集合
        self.forbidden_columns: Set[str] = set()
        
        # 操作约束：禁止执行的操作序列
        self.forbidden_operations: Set[Tuple[str, Tuple[str, ...]]] = set()
        
        # 聚合约束：强制修正的聚合作用域
        self.aggregation_constraints: Dict[str, str] = {}
        
        # 约束历史：记录约束的来源和推理轮次
        self.constraint_history: List[Dict[str, Any]] = []
        
        # 当前推理轮次
        self.current_round: int = 0
        
        # 约束统计
        self.stats = {
            "total_constraints_added": 0,
            "row_constraints": 0,
            "column_constraints": 0,
            "operation_constraints": 0,
            "aggregation_constraints": 0
        }
    
    def add_forbidden_rows(
        self, 
        rows: List[str], 
        error_type: str, 
        source_step: int
    ) -> None:
        """
        添加禁止的行约束
        
        Args:
            rows: 禁止的行列表，如 ["row 1", "row 3"]
            error_type: 错误类型，如 "row_error"
            source_step: 错误步骤编号
        """
        new_rows = [row for row in rows if row not in self.forbidden_rows]
        for row in new_rows:
            self.forbidden_rows.add(row)
        
        if new_rows:
            self.constraint_history.append({
                "round": self.current_round,
                "type": error_type,
                "constraint_type": "forbidden_rows",
                "value": new_rows,
                "source_step": source_step
            })
            self.stats["row_constraints"] += len(new_rows)
            self.stats["total_constraints_added"] += len(new_rows)
    
    def add_forbidden_columns(
        self, 
        columns: List[str], 
        error_type: str, 
        source_step: int
    ) -> None:
        """
        添加禁止的列约束
        
        Args:
            columns: 禁止的列列表，如 ["column_name", ...]
            error_type: 错误类型，如 "column_error"
            source_step: 错误步骤编号
        """
        new_columns = [col for col in columns if col not in self.forbidden_columns]
        for col in new_columns:
            self.forbidden_columns.add(col)
        
        if new_columns:
            self.constraint_history.append({
                "round": self.current_round,
                "type": error_type,
                "constraint_type": "forbidden_columns",
                "value": new_columns,
                "source_step": source_step
            })
            self.stats["column_constraints"] += len(new_columns)
            self.stats["total_constraints_added"] += len(new_columns)
    
    def add_forbidden_operation(
        self, 
        operation_name: str, 
        parameters: List[str], 
        error_type: str, 
        source_step: int
    ) -> None:
        """
        添加禁止的操作约束
        
        Args:
            operation_name: 操作名称，如 "select_row"
            parameters: 操作参数列表
            error_type: 错误类型
            source_step: 错误步骤编号
        """
        operation_key = (operation_name, tuple(sorted(parameters)))
        
        if operation_key not in self.forbidden_operations:
            self.forbidden_operations.add(operation_key)
            self.constraint_history.append({
                "round": self.current_round,
                "type": error_type,
                "constraint_type": "forbidden_operation",
                "value": (operation_name, parameters),
                "source_step": source_step
            })
            self.stats["operation_constraints"] += 1
            self.stats["total_constraints_added"] += 1
    
    def add_aggregation_constraint(
        self, 
        constraint_key: str, 
        constraint_value: str, 
        error_type: str, 
        source_step: int
    ) -> None:
        """
        添加聚合约束
        
        Args:
            constraint_key: 约束键，如 "group_column"
            constraint_value: 约束值，如正确的列名
            error_type: 错误类型
            source_step: 错误步骤编号
        """
        # 如果约束值发生变化，记录历史
        old_value = self.aggregation_constraints.get(constraint_key)
        if old_value != constraint_value:
            self.aggregation_constraints[constraint_key] = constraint_value
            self.constraint_history.append({
                "round": self.current_round,
                "type": error_type,
                "constraint_type": "aggregation",
                "value": {constraint_key: constraint_value},
                "source_step": source_step,
                "previous_value": old_value
            })
            self.stats["aggregation_constraints"] += 1
            self.stats["total_constraints_added"] += 1
    
    def is_row_forbidden(self, row: str) -> bool:
        """
        检查行是否被禁止
        
        Args:
            row: 行标识，如 "row 1"
            
        Returns:
            bool: 是否被禁止
        """
        return row in self.forbidden_rows
    
    def is_column_forbidden(self, column: str) -> bool:
        """
        检查列是否被禁止
        
        Args:
            column: 列名
            
        Returns:
            bool: 是否被禁止
        """
        return column in self.forbidden_columns
    
    def is_operation_forbidden(
        self, 
        operation_name: str, 
        parameters: List[str]
    ) -> bool:
        """
        检查操作是否被禁止
        
        Args:
            operation_name: 操作名称
            parameters: 操作参数列表
            
        Returns:
            bool: 是否被禁止
        """
        operation_key = (operation_name, tuple(sorted(parameters)))
        return operation_key in self.forbidden_operations
    
    def get_aggregation_constraint(self, constraint_key: str) -> str:
        """
        获取聚合约束
        
        Args:
            constraint_key: 约束键
            
        Returns:
            str: 约束值，如果不存在则返回None
        """
        return self.aggregation_constraints.get(constraint_key)
    
    def increment_round(self) -> None:
        """增加推理轮次"""
        self.current_round += 1
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """
        获取约束摘要
        
        Returns:
            Dict: 包含所有约束信息的字典
        """
        return {
            "forbidden_rows": sorted(list(self.forbidden_rows)),
            "forbidden_columns": sorted(list(self.forbidden_columns)),
            "forbidden_operations": [
                {"operation": op[0], "parameters": list(op[1])}
                for op in self.forbidden_operations
            ],
            "aggregation_constraints": self.aggregation_constraints,
            "total_constraints": self.stats["total_constraints_added"],
            "constraint_breakdown": self.stats,
            "current_round": self.current_round,
            "history_length": len(self.constraint_history)
        }
    
    def get_constraints_for_prompt(self) -> str:
        """
        生成用于prompt的约束描述
        
        Returns:
            str: 约束的自然语言描述
        """
        prompt_parts = []
        
        if self.forbidden_rows:
            rows_str = ", ".join(sorted(self.forbidden_rows))
            prompt_parts.append(f"DO NOT select the following rows: {rows_str}")
        
        if self.forbidden_columns:
            cols_str = ", ".join(sorted(self.forbidden_columns))
            prompt_parts.append(f"DO NOT select the following columns: {cols_str}")
        
        if self.forbidden_operations:
            op_parts = []
            for op_name, params in self.forbidden_operations:
                params_str = ", ".join(params)
                op_parts.append(f"{op_name}({params_str})")
            prompt_parts.append(f"DO NOT perform the following operations: {', '.join(op_parts)}")
        
        if self.aggregation_constraints:
            agg_parts = []
            for key, value in self.aggregation_constraints.items():
                agg_parts.append(f"{key} should be '{value}'")
            prompt_parts.append(f"Aggregation constraints: {'; '.join(agg_parts)}")
        
        if prompt_parts:
            return "\n".join(prompt_parts) + "\n"
        else:
            return "No constraints applied yet.\n"
    
    def merge(self, other: 'ConstraintState') -> None:
        """
        合并另一个约束状态
        
        Args:
            other: 另一个ConstraintState实例
        """
        self.forbidden_rows.update(other.forbidden_rows)
        self.forbidden_columns.update(other.forbidden_columns)
        self.forbidden_operations.update(other.forbidden_operations)
        self.aggregation_constraints.update(other.aggregation_constraints)
        self.constraint_history.extend(other.constraint_history)
        
        # 更新统计
        for key in self.stats:
            if key in other.stats:
                self.stats[key] += other.stats[key]
    
    def copy(self) -> 'ConstraintState':
        """
        创建约束状态的深拷贝
        
        Returns:
            ConstraintState: 拷贝的实例
        """
        new_state = ConstraintState()
        new_state.forbidden_rows = copy.deepcopy(self.forbidden_rows)
        new_state.forbidden_columns = copy.deepcopy(self.forbidden_columns)
        new_state.forbidden_operations = copy.deepcopy(self.forbidden_operations)
        new_state.aggregation_constraints = copy.deepcopy(self.aggregation_constraints)
        new_state.constraint_history = copy.deepcopy(self.constraint_history)
        new_state.current_round = self.current_round
        new_state.stats = copy.deepcopy(self.stats)
        return new_state
    
    def clear_round_constraints(self, round_num: int) -> None:
        """
        清除特定轮次的约束
        
        Args:
            round_num: 要清除的轮次编号
        """
        # 清除该轮次添加的约束
        for history_item in self.constraint_history:
            if history_item["round"] == round_num:
                constraint_type = history_item["constraint_type"]
                value = history_item["value"]
                
                if constraint_type == "forbidden_rows":
                    for row in value:
                        self.forbidden_rows.discard(row)
                elif constraint_type == "forbidden_columns":
                    for col in value:
                        self.forbidden_columns.discard(col)
                elif constraint_type == "forbidden_operation":
                    op_name, params = value
                    self.forbidden_operations.discard((op_name, tuple(sorted(params))))
                elif constraint_type == "aggregation":
                    for key in value.keys():
                        self.aggregation_constraints.pop(key, None)
        
        # 移除该轮次的历史记录
        self.constraint_history = [
            h for h in self.constraint_history 
            if h["round"] != round_num
        ]
    
    def __repr__(self) -> str:
        return f"ConstraintState(round={self.current_round}, constraints={self.stats['total_constraints_added']})"
