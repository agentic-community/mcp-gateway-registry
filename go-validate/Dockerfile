# Build a static, dependency-free binary and ship it on a minimal base.
# Build base uses the AWS ECR Public mirror of the Docker official golang image
# (same convention as docker/Dockerfile.metrics-db) to avoid Docker Hub rate
# limits. This is the BUILD stage only; nothing from it ships (multi-stage) -- the
# runtime image below is distroless/static (no shell, minimal CVE surface).
FROM public.ecr.aws/docker/library/golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/go-validate .

# Fetch the Amazon DocumentDB CA bundle at build time (same source the Python
# entrypoints use). DocumentDB serves a cert signed by the Amazon RDS CA, which is
# not a public root, so the scope-resolver's TLS connection must trust it. It is
# NOT vendored in git (*.pem is gitignored as a secret-guard).
RUN apk add --no-cache ca-certificates wget \
    && wget -O /out/global-bundle.pem https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/go-validate /go-validate
# CA bundle at the same path the Python containers use (DOCUMENTDB_TLS_CA_FILE default).
COPY --from=build /out/global-bundle.pem /app/certs/global-bundle.pem
EXPOSE 8899
USER nonroot:nonroot
ENTRYPOINT ["/go-validate"]
