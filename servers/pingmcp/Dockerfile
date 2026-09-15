FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/pingmcp .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/pingmcp /pingmcp
EXPOSE 8100
USER nonroot:nonroot
ENTRYPOINT ["/pingmcp"]
