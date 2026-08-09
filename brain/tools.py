import ast
import operator

def safe_calc(expression):
    """
    Safely evaluate a mathematical expression using ast.
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant): # python 3.8+
            return node.value
        elif type(node).__name__ == 'Num': # python <3.8 fallback
            return node.n
        elif isinstance(node, ast.BinOp): # <left> <operator> <right>
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
            return operators[type(node.op)](eval_node(node.operand))
        else:
            raise TypeError(node)

    try:
        # Parse the expression into an AST
        node = ast.parse(expression, mode='eval').body
        result = eval_node(node)
        
        # Format cleanly
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    except Exception as e:
        return f"Error: Invalid expression ({str(e)})"
