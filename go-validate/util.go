package main

import "strconv"

// strconvFormat formats an int64 as a decimal string.
func strconvFormat(n int64) string {
	return strconv.FormatInt(n, 10)
}
