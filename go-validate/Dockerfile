# Build a static, dependency-free binary and ship it on a minimal base.
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/go-validate .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/go-validate /go-validate
EXPOSE 8899
USER nonroot:nonroot
ENTRYPOINT ["/go-validate"]
