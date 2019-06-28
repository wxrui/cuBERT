/* cuBERT.i */
%module cuBERT
%{
/* Includes the header in the wrapper code */
#include "../src/cuBERT.h"
typedef signed char int8_t;
%}

%include "various.i"
%include "carrays.i"
%include "cpointer.i"
%include "arrays_java.i"

/* header files */
%include "../src/cuBERT.h"

%array_class(double, DoubleArray)
%array_class(float, FloatArray)
%array_class(int, IntArray)

%array_class(signed char, ByteArray)

/* Note: there is a bug in the SWIG generated string arrays when creating
   a new array with strings where the strings are prematurely deallocated
*/
%array_functions(char *, StringArray)