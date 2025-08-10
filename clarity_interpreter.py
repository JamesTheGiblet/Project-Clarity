#!/usr/bin/env python3
"""
Clarity Language Interpreter
A prototype interpreter for the Clarity programming language
"""

import re
import sys
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Token types for lexical analysis
class TokenType(Enum):
    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    
    # Identifiers and keywords
    IDENTIFIER = "IDENTIFIER"
    TYPE = "TYPE"
    LET = "LET"
    MUT = "MUT"
    FUNCTION = "FUNCTION"
    MATCH = "MATCH"
    WITH = "WITH"
    IF = "IF"
    ELSE = "ELSE"
    FOR = "FOR"
    IN = "IN"
    RETURN = "RETURN"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TEST = "TEST"
    ASSERT = "ASSERT"
    
    # Operators
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    ASSIGN = "ASSIGN"
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    ARROW = "ARROW"
    QUESTION = "QUESTION"
    
    # Delimiters
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COMMA = "COMMA"
    COLON = "COLON"
    SEMICOLON = "SEMICOLON"
    DOT = "DOT"
    
    # Special
    NEWLINE = "NEWLINE"
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        
        self.keywords = {
            'let': TokenType.LET,
            'mut': TokenType.MUT,
            'function': TokenType.FUNCTION,
            'match': TokenType.MATCH,
            'with': TokenType.WITH,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'for': TokenType.FOR,
            'in': TokenType.IN,
            'return': TokenType.RETURN,
            'Success': TokenType.SUCCESS,
            'Error': TokenType.ERROR,
            'test': TokenType.TEST,
            'assert': TokenType.ASSERT,
            'true': TokenType.BOOLEAN,
            'false': TokenType.BOOLEAN,
            'Number': TokenType.TYPE,
            'String': TokenType.TYPE,
            'Boolean': TokenType.TYPE,
            'List': TokenType.TYPE,
            'Result': TokenType.TYPE,
        }
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]
    
    def peek_char(self) -> Optional[str]:
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]
    
    def advance(self):
        if self.pos < len(self.text) and self.text[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '/' and self.peek_char() == '/':
            while self.current_char() and self.current_char() != '\n':
                self.advance()
    
    def read_number(self) -> Token:
        start_column = self.column
        num_str = ''
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            num_str += self.current_char()
            self.advance()
        
        return Token(TokenType.NUMBER, num_str, self.line, start_column)
    
    def read_string(self) -> Token:
        start_column = self.column
        quote_char = self.current_char()
        self.advance()  # Skip opening quote
        
        string_val = ''
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                escape_char = self.current_char()
                if escape_char == 'n':
                    string_val += '\n'
                elif escape_char == 't':
                    string_val += '\t'
                elif escape_char == '\\':
                    string_val += '\\'
                elif escape_char == quote_char:
                    string_val += quote_char
                else:
                    string_val += escape_char
            else:
                string_val += self.current_char()
            self.advance()
        
        if self.current_char() == quote_char:
            self.advance()  # Skip closing quote
        
        return Token(TokenType.STRING, string_val, self.line, start_column)
    
    def read_identifier(self) -> Token:
        start_column = self.column
        identifier = ''
        
        while (self.current_char() and 
               (self.current_char().isalnum() or self.current_char() in '_')):
            identifier += self.current_char()
            self.advance()
        
        token_type = self.keywords.get(identifier, TokenType.IDENTIFIER)
        return Token(token_type, identifier, self.line, start_column)
    
    def tokenize(self) -> List[Token]:
        tokens = []
        
        while self.current_char():
            self.skip_whitespace()
            
            if not self.current_char():
                break
            
            # Comments
            if self.current_char() == '/' and self.peek_char() == '/':
                self.skip_comment()
                continue
            
            # Newlines
            if self.current_char() == '\n':
                tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                continue
            
            # Numbers
            if self.current_char().isdigit():
                tokens.append(self.read_number())
                continue
            
            # Strings
            if self.current_char() in '"\'':
                tokens.append(self.read_string())
                continue
            
            # Identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                tokens.append(self.read_identifier())
                continue
            
            # Two-character operators
            if self.current_char() == '-' and self.peek_char() == '>':
                tokens.append(Token(TokenType.ARROW, '->', self.line, self.column))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '=' and self.peek_char() == '=':
                tokens.append(Token(TokenType.EQUALS, '==', self.line, self.column))
                self.advance()
                self.advance()
                continue
            
            if self.current_char() == '!' and self.peek_char() == '=':
                tokens.append(Token(TokenType.NOT_EQUALS, '!=', self.line, self.column))
                self.advance()
                self.advance()
                continue
            
            # Single-character operators and delimiters
            char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '=': TokenType.ASSIGN,
                '<': TokenType.LESS_THAN,
                '>': TokenType.GREATER_THAN,
                '?': TokenType.QUESTION,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
                '.': TokenType.DOT,
            }
            
            if self.current_char() in char_tokens:
                char = self.current_char()
                tokens.append(Token(char_tokens[char], char, self.line, self.column))
                self.advance()
                continue
            
            # Unknown character
            raise SyntaxError(f"Unknown character '{self.current_char()}' at line {self.line}, column {self.column}")
        
        tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return tokens

# Abstract Syntax Tree nodes
class ASTNode:
    pass

@dataclass
class NumberNode(ASTNode):
    value: float

@dataclass
class StringNode(ASTNode):
    value: str

@dataclass
class BooleanNode(ASTNode):
    value: bool

@dataclass
class IdentifierNode(ASTNode):
    name: str

@dataclass
class BinaryOpNode(ASTNode):
    left: ASTNode
    operator: str
    right: ASTNode

@dataclass
class UnaryOpNode(ASTNode):
    operator: str
    operand: ASTNode

@dataclass
class AssignmentNode(ASTNode):
    name: str
    value: ASTNode
    mutable: bool = False

@dataclass
class FunctionDefNode(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]

@dataclass
class FunctionCallNode(ASTNode):
    name: str
    args: List[ASTNode]

@dataclass
class MatchNode(ASTNode):
    expr: ASTNode
    cases: List[Tuple[str, ASTNode]]  # (pattern, body)

@dataclass
class ReturnNode(ASTNode):
    value: Optional[ASTNode]

@dataclass
class TestNode(ASTNode):
    name: str
    body: List[ASTNode]

@dataclass
class AssertNode(ASTNode):
    condition: ASTNode
    message: Optional[str] = None

@dataclass
class ListNode(ASTNode):
    elements: List[ASTNode]

@dataclass
class ResultNode(ASTNode):
    is_success: bool
    value: ASTNode

# Runtime values
class ClarityValue:
    pass

@dataclass
class ClarityNumber(ClarityValue):
    value: float
    
    def __str__(self):
        return str(int(self.value) if self.value.is_integer() else self.value)

@dataclass
class ClarityString(ClarityValue):
    value: str
    
    def __str__(self):
        return self.value

@dataclass
class ClarityBoolean(ClarityValue):
    value: bool
    
    def __str__(self):
        return "true" if self.value else "false"

@dataclass
class ClarityList(ClarityValue):
    elements: List[ClarityValue]
    
    def __str__(self):
        return f"[{', '.join(str(e) for e in self.elements)}]"

@dataclass
class ClarityResult(ClarityValue):
    is_success: bool
    value: ClarityValue
    
    def __str__(self):
        if self.is_success:
            return f"Success({self.value})"
        else:
            return f"Error({self.value})"

@dataclass
class ClarityFunction(ClarityValue):
    params: List[str]
    body: List[ASTNode]
    closure: Dict[str, ClarityValue]

class ClarityError(Exception):
    def __init__(self, message: str, value: ClarityValue = None):
        self.message = message
        self.value = value or ClarityString(message)
        super().__init__(message)

# Simple parser (recursive descent)
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current_token(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[self.pos]
    
    def advance(self):
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.current_token()
        if token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type} at line {token.line}")
        self.advance()
        return token
    
    def skip_newlines(self):
        while self.current_token().type == TokenType.NEWLINE:
            self.advance()
    
    def parse_program(self) -> List[ASTNode]:
        nodes = []
        self.skip_newlines()
        
        while self.current_token().type != TokenType.EOF:
            if self.current_token().type == TokenType.NEWLINE:
                self.advance()
                continue
            
            node = self.parse_statement()
            if node:
                nodes.append(node)
            self.skip_newlines()
        
        return nodes
    
    def parse_statement(self) -> Optional[ASTNode]:
        if self.current_token().type == TokenType.LET:
            return self.parse_assignment()
        elif self.current_token().type == TokenType.FUNCTION:
            return self.parse_function_def()
        elif self.current_token().type == TokenType.RETURN:
            return self.parse_return()
        elif self.current_token().type == TokenType.MATCH:
            return self.parse_match()
        elif self.current_token().type == TokenType.TEST:
            return self.parse_test()
        elif self.current_token().type == TokenType.ASSERT:
            return self.parse_assert()
        else:
            return self.parse_expression()
    
    def parse_assignment(self) -> AssignmentNode:
        self.expect(TokenType.LET)
        
        mutable = False
        if self.current_token().type == TokenType.MUT:
            mutable = True
            self.advance()
        
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        
        return AssignmentNode(name, value, mutable)
    
    def parse_function_def(self) -> FunctionDefNode:
        self.expect(TokenType.FUNCTION)
        name = self.expect(TokenType.IDENTIFIER).value
        
        self.expect(TokenType.LPAREN)
        params = []
        
        while self.current_token().type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENTIFIER).value)
            if self.current_token().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.COLON)
        self.skip_newlines()
        
        body = []
        while (self.current_token().type not in [TokenType.EOF, TokenType.FUNCTION, TokenType.TEST] and 
               not (self.current_token().type == TokenType.IDENTIFIER and self.pos > 0)):
            if self.current_token().type == TokenType.NEWLINE:
                self.advance()
                continue
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
            
            # Simple heuristic to stop parsing function body
            if (self.current_token().type == TokenType.IDENTIFIER and 
                self.pos + 1 < len(self.tokens) and 
                self.tokens[self.pos + 1].type in [TokenType.ASSIGN, TokenType.LPAREN]):
                break
        
        return FunctionDefNode(name, params, body)
    
    def parse_return(self) -> ReturnNode:
        self.expect(TokenType.RETURN)
        value = None
        if self.current_token().type not in [TokenType.NEWLINE, TokenType.EOF]:
            value = self.parse_expression()
        return ReturnNode(value)
    
    def parse_match(self) -> MatchNode:
        self.expect(TokenType.MATCH)
        expr = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()
        
        cases = []
        while self.current_token().type == TokenType.IDENTIFIER:
            pattern = self.current_token().value
            self.advance()
            self.expect(TokenType.ARROW)
            body = self.parse_expression()
            cases.append((pattern, body))
            self.skip_newlines()
        
        return MatchNode(expr, cases)
    
    def parse_test(self) -> TestNode:
        self.expect(TokenType.TEST)
        name = self.expect(TokenType.STRING).value
        self.expect(TokenType.COLON)
        self.skip_newlines()
        
        body = []
        while (self.current_token().type not in [TokenType.EOF, TokenType.TEST, TokenType.FUNCTION]):
            if self.current_token().type == TokenType.NEWLINE:
                self.advance()
                continue
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
            
            if (self.current_token().type == TokenType.IDENTIFIER and 
                self.pos + 1 < len(self.tokens) and 
                self.tokens[self.pos + 1].type in [TokenType.ASSIGN]):
                break
        
        return TestNode(name, body)
    
    def parse_assert(self) -> AssertNode:
        self.expect(TokenType.ASSERT)
        condition = self.parse_expression()
        message = None
        
        if self.current_token().type == TokenType.COMMA:
            self.advance()
            message = self.expect(TokenType.STRING).value
        
        return AssertNode(condition, message)
    
    def parse_expression(self) -> ASTNode:
        return self.parse_logical_or()
    
    def parse_logical_or(self) -> ASTNode:
        node = self.parse_logical_and()
        
        while self.current_token().type == TokenType.OR:
            op = self.current_token().value
            self.advance()
            right = self.parse_logical_and()
            node = BinaryOpNode(node, op, right)
        
        return node
    
    def parse_logical_and(self) -> ASTNode:
        node = self.parse_equality()
        
        while self.current_token().type == TokenType.AND:
            op = self.current_token().value
            self.advance()
            right = self.parse_equality()
            node = BinaryOpNode(node, op, right)
        
        return node
    
    def parse_equality(self) -> ASTNode:
        node = self.parse_comparison()
        
        while self.current_token().type in [TokenType.EQUALS, TokenType.NOT_EQUALS]:
            op = self.current_token().value
            self.advance()
            right = self.parse_comparison()
            node = BinaryOpNode(node, op, right)
        
        return node
    
    def parse_comparison(self) -> ASTNode:
        node = self.parse_addition()
        
        while self.current_token().type in [TokenType.LESS_THAN, TokenType.GREATER_THAN]:
            op = self.current_token().value
            self.advance()
            right = self.parse_addition()
            node = BinaryOpNode(node, op, right)
        
        return node
    
    def parse_addition(self) -> ASTNode:
        node = self.parse_multiplication()
        
        while self.current_token().type in [TokenType.PLUS, TokenType.MINUS]:
            op = self.current_token().value
            self.advance()
            right = self.parse_multiplication()
            node = BinaryOpNode(node, op, right)
        
        return node
    
    def parse_multiplication(self) -> ASTNode:
        node = self.parse_unary()
        
        while self.current_token().type in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            op = self.current_token().value
            self.advance()
            right = self.parse_unary()
            node = BinaryOpNode(node, op, right)
        
        return node
    
    def parse_unary(self) -> ASTNode:
        if self.current_token().type in [TokenType.MINUS, TokenType.NOT]:
            op = self.current_token().value
            self.advance()
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        token = self.current_token()
        
        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(float(token.value))
        
        elif token.type == TokenType.STRING:
            self.advance()
            return StringNode(token.value)
        
        elif token.type == TokenType.BOOLEAN:
            self.advance()
            return BooleanNode(token.value == 'true')
        
        elif token.type == TokenType.SUCCESS:
            self.advance()
            self.expect(TokenType.LPAREN)
            value = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return ResultNode(True, value)
        
        elif token.type == TokenType.ERROR:
            self.advance()
            self.expect(TokenType.LPAREN)
            value = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return ResultNode(False, value)
        
        elif token.type == TokenType.LBRACKET:
            self.advance()
            elements = []
            
            while self.current_token().type != TokenType.RBRACKET:
                elements.append(self.parse_expression())
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
            
            self.expect(TokenType.RBRACKET)
            return ListNode(elements)
        
        elif token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()
            
            # Function call
            if self.current_token().type == TokenType.LPAREN:
                self.advance()
                args = []
                
                while self.current_token().type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    if self.current_token().type == TokenType.COMMA:
                        self.advance()
                
                self.expect(TokenType.RPAREN)
                return FunctionCallNode(name, args)
            
            # Variable reference
            return IdentifierNode(name)
        
        elif token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        else:
            raise SyntaxError(f"Unexpected token {token.type} at line {token.line}")

# Interpreter
class Environment:
    def __init__(self, parent: Optional['Environment'] = None):
        self.vars: Dict[str, ClarityValue] = {}
        self.parent = parent
    
    def get(self, name: str) -> ClarityValue:
        if name in self.vars:
            return self.vars[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NameError(f"Variable '{name}' not defined")
    
    def set(self, name: str, value: ClarityValue):
        self.vars[name] = value
    
    def define(self, name: str, value: ClarityValue):
        self.vars[name] = value

class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        self.setup_builtins()
    
    def setup_builtins(self):
        # Built-in print function
        def builtin_print(*args):
            values = []
            for arg in args:
                if isinstance(arg, ClarityValue):
                    values.append(str(arg))
                else:
                    values.append(str(arg))
            print(' '.join(values))
            return ClarityString(' '.join(values))
        
        self.global_env.define('print', builtin_print)
    
    def interpret(self, nodes: List[ASTNode]) -> Optional[ClarityValue]:
        result = None
        for node in nodes:
            result = self.evaluate(node, self.global_env)
            if isinstance(node, TestNode):
                self.run_test(node)
        return result
    
    def run_test(self, test_node: TestNode):
        print(f"Running test: {test_node.name}")
        test_env = Environment(self.global_env)
        
        try:
            for stmt in test_node.body:
                self.evaluate(stmt, test_env)
            print(f"✓ Test '{test_node.name}' passed")
        except Exception as e:
            print(f"✗ Test '{test_node.name}' failed: {e}")
    
    def evaluate(self, node: ASTNode, env: Environment) -> ClarityValue:
        if isinstance(node, NumberNode):
            return ClarityNumber(node.value)
        
        elif isinstance(node, StringNode):
            return ClarityString(node.value)
        
        elif isinstance(node, BooleanNode):
            return ClarityBoolean(node.value)
        
        elif isinstance(node, IdentifierNode):
            return env.get(node.name)
        
        elif isinstance(node, ListNode):
            elements = [self.evaluate(elem, env) for elem in node.elements]
            return ClarityList(elements)
        
        elif isinstance(node, ResultNode):
            value = self.evaluate(node.value, env)
            return ClarityResult(node.is_success, value)
        
        elif isinstance(node, BinaryOpNode):
            return self.evaluate_binary_op(node, env)
        
        elif isinstance(node, UnaryOpNode):
            return self.evaluate_unary_op(node, env)
        
        elif isinstance(node, AssignmentNode):
            value = self.evaluate(node.value, env)
            env.define(node.name, value)
            return value
        
        elif isinstance(node, FunctionDefNode):
            func = ClarityFunction(node.params, node.body, env.vars.copy())
            env.define(node.name, func)
            return func
        
        elif isinstance(node, FunctionCallNode):
            return self.call_function(node, env)
        
        elif isinstance(node, MatchNode):
            return self.evaluate_match(node, env)
        
        elif isinstance(node, AssertNode):
            condition = self.evaluate(node.condition, env)
            if not self.is_truthy(condition):
                message = node.message or "Assertion failed"
                raise AssertionError(message)
            return ClarityBoolean(True)
        
        elif isinstance(node, ReturnNode):
            if node.value:
                return self.evaluate(node.value, env)
            return ClarityString("")
        
        else:
            raise NotImplementedError(f"Evaluation not implemented for {type(node)}")
    
    def evaluate_binary_op(self, node: BinaryOpNode, env: Environment) -> ClarityValue:
        left = self.evaluate(node.left, env)
        right = self.evaluate(node.right, env)
        
        if node.operator == '+':
            if isinstance(left, ClarityNumber) and isinstance(right, ClarityNumber):
                return ClarityNumber(left.value + right.value)
            elif isinstance(left, ClarityString) and isinstance(right, ClarityString):
                return ClarityString(left.value + right.value)
        
        elif node.operator == '-':
            if isinstance(left, ClarityNumber) and isinstance(right, ClarityNumber):
                return ClarityNumber(left.value - right.value)
        
        elif node.operator == '*':
            if isinstance(left, ClarityNumber) and isinstance(right, ClarityNumber):
                return ClarityNumber(left.value * right.value)
        
        elif node.operator == '/':
            if isinstance(left, ClarityNumber) and isinstance(right, ClarityNumber):
                if right.value == 0:
                    raise ZeroDivisionError("Division by zero")
                return ClarityNumber(left.value / right.value)
        
        elif node.operator == '==':
            return ClarityBoolean(self.values_equal(left, right))
        
        elif node.operator == '!=':
            return ClarityBoolean(not self.values_equal(left, right))
        
        elif node.operator == '<':
            if isinstance(left, ClarityNumber) and isinstance(right, ClarityNumber):
                return ClarityBoolean(left.value < right.value)
        
        elif node.operator == '>':
            if isinstance(left, ClarityNumber) and isinstance(right, ClarityNumber):
                return ClarityBoolean(left.value > right.value)
        
        raise TypeError(f"Unsupported operation: {type(left)} {node.operator} {type(right)}")
    
    def evaluate_unary_op(self, node: UnaryOpNode, env: Environment) -> ClarityValue:
        operand = self.evaluate(node.operand, env)
        
        if node.operator == '-':
            if isinstance(operand, ClarityNumber):
                return ClarityNumber(-operand.value)
        
        elif node.operator == 'not':
            return ClarityBoolean(not self.is_truthy(operand))
        
        raise TypeError(f"Unsupported unary operation: {node.operator} {type(operand)}")
    
    def call_function(self, node: FunctionCallNode, env: Environment) -> ClarityValue:
        func = env.get(node.name)
        
        # Built-in function
        if callable(func):
            args = [self.evaluate(arg, env) for arg in node.args]
            return func(*args)
        
        # User-defined function
        elif isinstance(func, ClarityFunction):
            if len(node.args) != len(func.params):
                raise TypeError(f"Function {node.name} expects {len(func.params)} arguments, got {len(node.args)}")
            
            # Create new environment for function execution
            func_env = Environment(env)
            
            # Bind parameters to arguments
            for param, arg_node in zip(func.params, node.args):
                arg_value = self.evaluate(arg_node, env)
                func_env.define(param, arg_value)
            
            # Execute function body
            result = None
            for stmt in func.body:
                result = self.evaluate(stmt, func_env)
                if isinstance(stmt, ReturnNode):
                    break
            
            return result if result else ClarityString("")
        
        else:
            raise TypeError(f"'{node.name}' is not a function")
    
    def evaluate_match(self, node: MatchNode, env: Environment) -> ClarityValue:
        value = self.evaluate(node.expr, env)
        
        # Simple pattern matching - just match on type/value for now
        for pattern, body in node.cases:
            if self.pattern_matches(pattern, value):
                return self.evaluate(body, env)
        
        raise ValueError(f"No matching pattern for {value}")
    
    def pattern_matches(self, pattern: str, value: ClarityValue) -> bool:
        # Very simple pattern matching
        if pattern == "_":  # Wildcard
            return True
        elif pattern == "Success" and isinstance(value, ClarityResult):
            return value.is_success
        elif pattern == "Error" and isinstance(value, ClarityResult):
            return not value.is_success
        elif isinstance(value, ClarityNumber):
            try:
                return float(pattern) == value.value
            except ValueError:
                return False
        elif isinstance(value, ClarityString):
            return pattern.strip('"\'') == value.value
        elif isinstance(value, ClarityBoolean):
            return pattern.lower() == str(value.value).lower()
        
        return False
    
    def values_equal(self, left: ClarityValue, right: ClarityValue) -> bool:
        if type(left) != type(right):
            return False
        
        if isinstance(left, ClarityNumber):
            return left.value == right.value
        elif isinstance(left, ClarityString):
            return left.value == right.value
        elif isinstance(left, ClarityBoolean):
            return left.value == right.value
        elif isinstance(left, ClarityList):
            if len(left.elements) != len(right.elements):
                return False
            return all(self.values_equal(l, r) for l, r in zip(left.elements, right.elements))
        
        return False
    
    def is_truthy(self, value: ClarityValue) -> bool:
        if isinstance(value, ClarityBoolean):
            return value.value
        elif isinstance(value, ClarityNumber):
            return value.value != 0
        elif isinstance(value, ClarityString):
            return len(value.value) > 0
        elif isinstance(value, ClarityList):
            return len(value.elements) > 0
        else:
            return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python clarity_interpreter.py <filename>")
        return
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            source_code = f.read()
        
        # Tokenize
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse_program()
        
        # Interpret
        interpreter = Interpreter()
        interpreter.interpret(ast)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except Exception as e:
        print(f"Error: {e}")

# Simple REPL for interactive testing
def repl():
    interpreter = Interpreter()
    print("Clarity REPL - Type 'exit' to quit")
    
    while True:
        try:
            line = input("clarity> ").strip()
            
            if line == 'exit':
                break
            
            if not line:
                continue
            
            # Tokenize and parse single expression/statement
            lexer = Lexer(line)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            
            # Try to parse as single statement
            if tokens and tokens[0].type in [TokenType.LET, TokenType.FUNCTION, TokenType.TEST]:
                ast = parser.parse_program()
            else:
                # Parse as expression
                expr = parser.parse_expression()
                ast = [expr]
            
            result = interpreter.interpret(ast)
            if result and not isinstance(result, type(lambda: None)):
                print(f"=> {result}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        repl()
    else:
        main()

# Example Clarity programs to test with:

example_programs = {
    "hello.cl": '''
// Simple hello world
let message = "Hello, Clarity!"
print(message)
''',

    "math.cl": '''
// Basic math operations
let x = 10
let y = 5
let sum = x + y
let product = x * y

print("Sum:", sum)
print("Product:", product)

function add(a, b):
    return a + b

let result = add(20, 30)
print("Function result:", result)
''',

    "result_types.cl": '''
// Result type handling
function divide(a, b):
    if b == 0:
        return Error("Division by zero")
    else:
        return Success(a / b)

let result1 = divide(10, 2)
let result2 = divide(10, 0)

print("Result 1:", result1)
print("Result 2:", result2)

match result1:
    Success -> print("Division succeeded")
    Error -> print("Division failed")
    _ -> print("Unknown result")
''',

    "lists.cl": '''
// List operations
let numbers = [1, 2, 3, 4, 5]
print("Numbers:", numbers)

let empty_list = []
print("Empty list:", empty_list)
''',

    "tests.cl": '''
// Testing example
function square(x):
    return x * x

test "square function":
    assert square(4) == 16
    assert square(0) == 0
    assert square(-3) == 9

test "basic arithmetic":
    assert 2 + 2 == 4
    assert 10 - 5 == 5
    assert 3 * 4 == 12
'''
}

# Create example files for testing
def create_examples():
    import os
    
    examples_dir = "clarity_examples"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    
    for filename, content in example_programs.items():
        filepath = os.path.join(examples_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content.strip())
    
    print(f"Created example files in {examples_dir}/")
    print("Run them with: python clarity_interpreter.py clarity_examples/<filename>")

# Uncomment to create example files
# create_examples()
