#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from ml.param.base_param import BaseParam


class EncryptedModeCalculatorParam(BaseParam):
    """
    Define the encrypted_mode_calulator parameters.

    Parameters
    ----------
    mode: str, support 'strict', 'fast', 'balance' only, default: strict

    re_encrypted_rate: float or int, numeric number, use when mode equals to 'strict', defualt: 1

    """

    def __init__(self, mode="strict", re_encrypted_rate=1):
        self.mode = mode
        self.re_encrypted_rate = re_encrypted_rate

    def check(self):
        descr = "encrypted_mode_calculator param"
        self.mode = self.check_and_change_lower(self.mode,
                                                ["strict", "fast", "balance"],
                                                descr)

        if self.mode == "balance":
            if type(self.re_encrypted_rate).__name__ not in ["int", "long", "float"]:
                raise ValueError("re_encrypted_rate should be a numeric number")

        return True

