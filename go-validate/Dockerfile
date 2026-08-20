# Build a static, dependency-free binary and ship it on a minimal base.
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/go-validate .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/go-validate /go-validate
# Amazon DocumentDB CA bundle: DocumentDB serves a cert signed by the Amazon RDS
# CA (not a public root), so the scope-resolver's TLS connection needs it. Baked
# in at the same path the Python containers use (DOCUMENTDB_TLS_CA_FILE default).
COPY global-bundle.pem /app/certs/global-bundle.pem
EXPOSE 8899
USER nonroot:nonroot
ENTRYPOINT ["/go-validate"]
