package crypto

import (
	"encoding/base64"
)

func Base64Encode(pwd string) string {
	return base64.StdEncoding.EncodeToString([]byte(pwd))
}
