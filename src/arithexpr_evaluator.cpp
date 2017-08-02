#include "arithexpr_evaluator.h"

#include "exprtk.hpp"

typedef exprtk::symbol_table<double> symbol_table_t;
typedef exprtk::expression<double> expression_t;
typedef exprtk::parser<double> parser_t;

// todo, do I need to make some of this stuff static for performance?
int Evaluator::evaluate(std::string expr, int v_MWidth_1, int v_MHeight_2,
                        int v_VLength_3) {
  symbol_table_t symbol_table;
  parser_t parser;
  expression_t expression;

  double dv_MWidth_1 = static_cast<double>(v_MWidth_1);
  double dv_MHeight_2 = static_cast<double>(v_MHeight_2);
  double dv_VLength_3 = static_cast<double>(v_VLength_3);

  symbol_table.add_variable("v_MWidthC_1", dv_MWidth_1);
  symbol_table.add_variable("v_MHeight_2", dv_MHeight_2);
  symbol_table.add_variable("v_VLength_3", dv_VLength_3);

  expression.register_symbol_table(symbol_table);
  parser.compile(expr, expression);
  double result = expression.value();

  return static_cast<int>(result);
}