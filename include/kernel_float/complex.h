#ifndef KERNEL_FLOAT_COMPLEX_TYPE_H
#define KERNEL_FLOAT_COMPLEX_TYPE_H

#include "macros.h"
#include "meta.h"

namespace kernel_float {

template<typename T>
struct alignas(2 * alignof(T)) complex_type_storage {
    T re;
    T im;
};

template<typename T>
struct complex_type: complex_type_storage<T> {
    using base_type = complex_type_storage<T>;

    template<typename T2>
    KERNEL_FLOAT_INLINE complex_type(complex_type<T2> that) : base_type(that.real(), that.imag()) {}

    KERNEL_FLOAT_INLINE
    complex_type(T real = {}, T imag = {}) : base_type(real, imag) {}

    KERNEL_FLOAT_INLINE
    T real() const {
        return this->re;
    }

    KERNEL_FLOAT_INLINE
    T imag() const {
        return this->im;
    }

    KERNEL_FLOAT_INLINE
    T norm() const {
        return real() * real() + imag() * imag();
    }

    KERNEL_FLOAT_INLINE
    complex_type conj() const {
        return {real(), -imag()};
    }
};

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> v) {
    return v;
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> a, complex_type<T> b) {
    return {a.real() + b.real(), a.imag() + b.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(T a, complex_type<T> b) {
    return {a + b.real(), b.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> a, T b) {
    return {a.real() + b, a.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator+=(complex_type<T>& a, complex_type<T> b) {
    return (a = a + b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator+=(complex_type<T>& a, T b) {
    return (a = a + b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> v) {
    return {-v.real(), -v.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> a, complex_type<T> b) {
    return {a.real() - b.real(), a.imag() - b.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(T a, complex_type<T> b) {
    return {a - b.real(), -b.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> a, T b) {
    return {a.real() - b, a.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator-=(complex_type<T>& a, complex_type<T> b) {
    return (a = a - b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator-=(complex_type<T>& a, T b) {
    return (a = a - b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(complex_type<T> a, complex_type<T> b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(complex_type<T> a, T b) {
    return {a.real() * b, a.imag() * b};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>* operator*=(complex_type<T>& a, complex_type<T> b) {
    return (a = a * b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator*=(complex_type<T>& a, T b) {
    return (a = a * b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(T a, complex_type<T> b) {
    return {a * b.real(), a * b.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(complex_type<T> a, complex_type<T> b) {
    T normi = T(1) / b.norm();

    return {
        (a.real() * b.real() + a.imag() * b.imag()) * normi,
        (a.imag() * b.real() - a.real() * b.imag()) * normi};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(complex_type<T> a, T b) {
    return a * (T(1) / b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(T a, complex_type<T> b) {
    T normi = T(1) / b.norm();

    return {a * b.real() * normi, -a * b.imag() * normi};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>* operator/=(complex_type<T>& a, complex_type<T> b) {
    return (a = a / b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator/=(complex_type<T>& a, T b) {
    return (a = a / b);
}

template<typename T>
KERNEL_FLOAT_INLINE T real(complex_type<T> v) {
    return v.real();
}

template<typename T>
KERNEL_FLOAT_INLINE T imag(complex_type<T> v) {
    return v.imag();
}

template<typename T>
KERNEL_FLOAT_INLINE T abs(complex_type<T> v) {
    return hypot(v.real(), v.imag());
}

template<typename T>
KERNEL_FLOAT_INLINE T arg(complex_type<T> v) {
    return atan2(v.imag(), v.real());
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> sqrt(complex_type<T> v) {
    T radius = abs(v);
    T cosA = v.real() / radius;

    complex_type<T> out = {
        sqrt(radius * (cosA + T(1)) * T(.5)),
        sqrt(radius * (T(1) - cosA) * T(.5))};

    // signbit should be false if x.y is negative
    if (v.imag() < 0) {
        out = complex_type<T> {out.real, -out.im};
    }

    return out;
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> norm(complex_type<T> v) {
    return v.real() * v.real() + v.imag() * v.imag();
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> conj(complex_type<T> v) {
    return {v.real(), -v.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> exp(complex_type<T> v) {
    // TODO: Handle nan and inf correctly
    T e = exp(v.real());
    T a = v.imag();
    return complex_type<T>(e * cos(a), e * sin(a));
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> log(complex_type<T> v) {
    return {log(abs(v)), arg(v)};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> pow(complex_type<T> a, T b) {
    return exp(a * log(b));
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> pow(complex_type<T> a, complex_type<T> b) {
    return exp(a * log(b));
}

template<typename L, typename R>
struct promote_type<complex_type<L>, complex_type<R>> {
    using type = complex_type<promote_t<L, R>>;
};

template<typename L, typename R>
struct promote_type<complex_type<L>, R> {
    using type = complex_type<promote_t<L, R>>;
};

template<typename L, typename R>
struct promote_type<L, complex_type<R>> {
    using type = complex_type<promote_t<L, R>>;
};

}  // namespace kernel_float

#endif
