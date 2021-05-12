/* mbed Microcontroller Library
 * Copyright (c) 2006-2013 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mbed.h"
#include "Arguments.h"
#include "pinmap.h"
#include "PinNames.h"

using namespace std;

namespace mbed {

PinName parse_pins(const char *str) {
#if defined(TARGET_LPC1768) || defined(TARGET_LPC11U24) || defined(TARGET_LPC2368)
    static const PinName pin_names[] = {p5, p6, p7, p8, p9, p10, p11, p12, p13, p14
                                , p15, p16, p17, p18, p19, p20, p21, p22, p23
                                , p24, p25, p26, p27, p28, p29, p30};
#elif defined(TARGET_LPC1114)
    static const PinName pin_names[] = {dp1, dp2, dp4, dp5, dp6, dp9, dp10, dp11
                                , dp13, dp14, dp15, dp16, dp17, dp18, dp23
                                , dp24, dp25, dp26, dp27, dp28};
#elif defined(TARGET_LPC4088)
    static const PinName pin_names[] = {NC, NC, NC, NC, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14
                                , p15, p16, p17, p18, p19, p20, NC, NC, p23
                                , p24, p25, p26, p27, p28, p29, p30, p31, p32
                                , p33, p34, NC, NC, p37, p38, p39, NC, NC, NC, NC, NC, NC, NC};
#elif defined(TARGET_LPC4088_DM)
    static const PinName pin_names[] = {p1, p2, p3, p4, NC, NC, p7, p8, p9, p10, p11, p12, p13, p14
                                , p15, p16, p17, p18, p19, p20, p21, p22, p23
                                , p24, p25, p26, NC, NC, p29, p30, NC, NC
                                , NC, NC, NC, NC, NC, NC, NC, NC, p41, p42, p43, p44, p45, p46};
#endif

#if defined(TARGET_LPC1768) || defined(TARGET_LPC11U24) || defined(TARGET_LPC2368) || defined(TARGET_LPC812) || defined(TARGET_LPC4088) || defined(TARGET_LPC4088_DM) || defined(TARGET_LPC1114) || defined(TARGET_RZ_A1H) || defined(TARGET_VK_RZ_A1H)
    if (str[0] == 'P') {              // Pn_n
        uint32_t port = str[1] - '0';
        uint32_t pin  = str[3] - '0'; // Pn_n
        uint32_t pin2 = str[4] - '0'; // Pn_nn
        if (pin2 <= 9) {
            pin = pin * 10 + pin2;
        }
        return port_pin((PortName)port, pin);

#elif defined(TARGET_KL25Z) || defined(TARGET_KL27Z) || defined(TARGET_KL05Z) || defined(TARGET_KL46Z) || defined(TARGET_K22F) || defined(TARGET_K64F) || defined(TARGET_K66F)
    if (str[0] == 'P' && str[1] == 'T') {   // PTxn
        uint32_t port = str[2] - 'A';
        uint32_t pin  = str[3] - '0';       // PTxn
        uint32_t pin2 = str[4] - '0';       // PTxnn

        if (pin2 <= 9) {
            pin = pin * 10 + pin2;
        }
        return port_pin((PortName)port, pin);

#elif defined(TARGET_NUCLEO_F072RB) || defined(TARGET_NUCLEO_F303K8) || defined(TARGET_NUCLEO_F411RE) || defined(TARGET_B_L4S5I_IOT01A)
    if (str[0] == 'P') {   // PX_XX e.g.PA_2 PC_15
        uint32_t port = str[1] - 'A';
        uint32_t pin  = str[3] - '0';       
        uint32_t pin2 = str[4] - '0';       

        if (pin2 <= 9) {
            pin = pin * 10 + pin2;
        }
        return port_pin((PortName)port, pin);

#endif



#if defined(TARGET_LPC1768) || defined(TARGET_LPC11U24) || defined(TARGET_LPC2368)
    } else if (str[0] == 'p') {       // pn
        uint32_t pin  = str[1] - '0'; // pn
        uint32_t pin2 = str[2] - '0'; // pnn
        if (pin2 <= 9) {
            pin = pin * 10 + pin2;
        }
        if (pin < 5 || pin > 30) {
            return NC;
        }
        return pin_names[pin - 5];
#elif defined(TARGET_LPC4088) || defined(TARGET_LPC4088_DM)
    } else if (str[0] == 'p') {       // pn
        uint32_t pin  = str[1] - '0'; // pn
        uint32_t pin2 = str[2] - '0'; // pnn
        if (pin2 <= 9) {
            pin = pin * 10 + pin2;
        }
        if (pin < 1 || pin > 46) {
            return NC;
        }
        return pin_names[pin - 1];
#endif
    } else if (str[0] == 'L') {  // LEDn
        switch (str[3]) {
            case '1' : return LED1;
            case '2' : return LED2;
            case '3' : return LED3;
        }

    } else if (str[0] == 'U') {  // USB?X
        switch (str[3]) {
            case 'T' : return USBTX;
            case 'R' : return USBRX;
        }
    }

    return NC;
}


Arguments::Arguments(const char* rqs) {
    obj_name = NULL;
    method_name = NULL;
    argc = 0;

    // This copy can be removed if we can assume the request string is
    // persistent and writable for the duration of the call
    strcpy(request, rqs);

    // Initial '/'
    char* p = request;
    if (*p != '/') return;
    p++;

    // Object Name
    p = search_arg(&obj_name, p, '/');
    if (p == NULL) return;

    // Method Name
    p = search_arg(&method_name, p, ' ');
    if (p == NULL) return;

    // Arguments
    while (true) {
        argv[argc] = NULL;
        p = search_arg(&argv[argc], p, ' ');
        if (argv[argc] != NULL) argc++;
        if (p == NULL) break;
    }

    index = -1;
}

char* Arguments::search_arg(char **arg, char *p, char next_sep) {
    char *s = p;
    while (true) {
        if ((*p == '/') || (*p == ' ') || (*p == '\n') || (*p == '\0')) break;
        p++;
    }
    if (p == s) return NULL;
    *arg = s;
    char separator = *p;
    *p = '\0';
    p++;
    return (separator == next_sep) ? (p) : (NULL);
}

template<> PinName Arguments::getArg<PinName>(void) {
    index++;
    return parse_pins(argv[index]);
}

template<> int Arguments::getArg<int>(void) {
    index++;
    char *pEnd;
    return strtol(argv[index], &pEnd, 10);
}

template<> const char* Arguments::getArg<const char*>(void) {
    index++;
    return argv[index];
}

template<> char Arguments::getArg<char>(void) {
    index++;
    return *argv[index];
}

template<> double Arguments::getArg<double>(void) {
    index++;
    return atof(argv[index]);
}

template<> float Arguments::getArg<float>(void) {
    index++;
    return atof(argv[index]);
}

Reply::Reply(char* r) {
    first = true;
    *r = '\0';
    reply = r;
}

void Reply::separator(void) {
    if (first) {
        first = false;
    } else {
        *reply = ' '; reply++;
    }
}

template<> void Reply::putData<const char*>(const char* s) {
    separator();
    reply += sprintf(reply, "%s", s);
}

template<> void Reply::putData<char*>(char* s) {
    separator();
    reply += sprintf(reply, "%s", s);
}

template<> void Reply::putData<char>(char c) {
    separator();
    reply += sprintf(reply, "%c", c);
}

template<> void Reply::putData<int>(int v) {
    separator();
    reply += sprintf(reply, "%d", v);
}

template<> void Reply::putData<float>(float f) {
    separator();
    reply += sprintf(reply, "%.17g", f);
}

} // namespace mbed
