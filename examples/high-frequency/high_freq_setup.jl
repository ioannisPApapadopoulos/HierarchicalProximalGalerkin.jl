k = 10π
c = 2 * k^2
f(x) = c* sin(k*x)
x0 = 0.0382764753038300
a0 = -22.6216651405434
x1 = 0.05
x2 = 0.85
x3 = 0.853422907268890
a3 = 6.74353407719214
b3 = -6.74353407719214

# k = 5π
# c = 2 * k^2
# f(x) = c* sin(k*x)
# x0 = 0.0765529506076600
# a0 = -11.3108325702717
# x1 = 0.1 
# x2 = 0.9
# x3 = 0.923447049392340
# a3 = 11.3108325702717
# b3 = -11.3108325702717
"""
FindRoot[{c/k * Cos[k*x] + a, c/k*Cos[k*y] + a, c/k^2*Sin[k*x] + a*x + b - 1, c/k^2*Sin[k*y] + a*y + b - 1}, {{x, 0.005}, {y, 0.005}, {a, 0}, {b, 0}}, WorkingPrecision -> 15]
FindRoot[{c/k * Cos[k*x] + a, c/k*Cos[k*y] + a, c/k^2*Sin[k*x] + a*x + b - 1, c/k^2*Sin[k*y] + a*y + b - 1}, {{x, 0.995}, {y, 0.995}, {a, 0}, {b, 0}}, WorkingPrecision -> 15]
FindRoot[{c/k * Cos[k*x] + a,  c/k^2*Sin[k*x] + a*x - 1}, {{x, 0.004}, {a, 0}}, WorkingPrecision -> 15]
FindRoot[{c/k * Cos[k*x] + a,  c/k^2*Sin[k*x]+a*x+b-1, a+b},{{x,0.988},{a,0}, {b,0}},WorkingPrecision->15]
"""

path = "output/1d_high_freq/"
if !isdir(path)
    mkpath(path)
end

φ(x) = 1.0
function ua(x::T) where T
    if x < x0
        return c / k^2 * sin(k*x) + a0*x
    elseif x0 ≤ x ≤ x1
        return 1.0
    elseif x1 ≤ x ≤ x2
        return c / k^2 * sin(k*x) - 1
    elseif x2 < x < x3
        return 1.0
    else
        return c / k^2 * sin(k*x) + a3*x + b3
    end
end

function save_data(ndofs, tics, avg_tics, h1s, subpath)
    writedlm(path*subpath*"_ndofs.log", ndofs)
    writedlm(path*subpath*"_avg_tics.log", avg_tics)
    writedlm(path*subpath*"_tics.log", tics)
    writedlm(path*subpath*"_h1s.log", h1s)
end
