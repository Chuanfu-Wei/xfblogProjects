function a(n, e) {
    for (var t = [1, 0, 2], i = 0; ;) {
        switch (t[i++]) {
            case 0:
                for (var a = 0, c = n.length; a < c; a++)
                    if (n[a] === e)
                        return !0;
                continue;
            case 1:
                if (n[["include", "s"].join("")])
                    return n.includes(e);
                continue;
            case 2:
                return !1
        }
        break
    }
}

let f1 = function (n) {
    for (var e = "pus", r = [], o = 0; o < n["length"]; o++) {
        var c = n.charCodeAt(o);
        if (32 === o)
            break;
        for (; a(r, c % 32);)
            c++;
        r[[e, "h"].join("")](c % 32)
    }
    return r
}
