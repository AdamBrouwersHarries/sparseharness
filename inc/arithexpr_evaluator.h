#pragma once

#include "csds_timer.h"
#include <cstdio>
#include <iostream>
#include <string>

class Evaluator {
public:
  // static void initialise_variables(int v_MWidth_1, int v_MHeight_2,
  //                                  int v_VLength_3);
  static int evaluate(std::string expr, int v_MWidth_1, int v_MHeight_2,
                      int v_VLength_3);
};