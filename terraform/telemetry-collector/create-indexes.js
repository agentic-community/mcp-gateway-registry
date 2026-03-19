use telemetry;

// TTL indexes (auto-delete after 365 days)
db.startup_events.createIndex(
  { "received_at": 1 },
  { expireAfterSeconds: 31536000 }
);

db.heartbeat_events.createIndex(
  { "received_at": 1 },
  { expireAfterSeconds: 31536000 }
);

// Query indexes
db.startup_events.createIndex({ "instance_id": 1 });
db.startup_events.createIndex({ "v": 1, "received_at": -1 });
db.heartbeat_events.createIndex({ "instance_id": 1 });

// Verify indexes
print("\n=== Startup Events Indexes ===");
printjson(db.startup_events.getIndexes());

print("\n=== Heartbeat Events Indexes ===");
printjson(db.heartbeat_events.getIndexes());

// Show stored events
print("\n=== Stored Startup Events ===");
print("Count: " + db.startup_events.count());
db.startup_events.find().forEach(printjson);

