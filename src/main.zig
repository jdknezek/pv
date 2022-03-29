const std = @import("std");
const argsParser = @import("zig-args");
const glob = @import("glob");

var allocator: std.mem.Allocator = undefined;

pub fn main() !u8 {
    const stdout = std.io.getStdOut().writer();
    var stderr = std.io.bufferedWriter(std.io.getStdErr().writer());

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    allocator = gpa.allocator();

    const args = try argsParser.parseForCurrentProcess(struct {
        buffer: u6 = 0,
        fps: u64 = 10,
        window: u64 = 60,
        help: bool = false,

        pub const shorthands = .{
            .b = "buffer",
            .f = "fps",
            .w = "window",
            .h = "help",
        };
    }, allocator, .print);
    defer args.deinit();

    if (args.options.help) {
        try stdout.print("Usage: {s} [options] globs...\n\nOptions:\n", .{std.fs.path.basename(args.executable_name.?)});
        try stdout.writeAll("  --buffer, -b  Buffer size as power of two (e.g. 20 = 1MiB) (default: 0 - autodetect)\n");
        try stdout.writeAll("  --fps, -f     Progress updates per second (default: 10)\n");
        try stdout.writeAll("  --window, -w  ETA window in seconds (default: 60)\n");
        return 1;
    }

    var dir = try std.fs.cwd().openDir(".", .{
        .access_sub_paths = true,
        .iterate = true,
    });
    defer dir.close();

    var files = std.ArrayList(FileInfo).init(allocator);
    defer {
        for (files.items) |info| {
            allocator.free(info.path);
            info.file.close();
        }
        files.deinit();
    }

    var total_size: usize = 0;

    for (args.positionals) |pattern| {
        var globber = try glob.Iterator.init(allocator, dir, pattern);
        defer globber.deinit();

        while (try globber.next()) |path| {
            const file = try dir.openFile(path, .{ .mode = .read_only });
            errdefer file.close();

            const stat = try file.stat();

            try files.append(.{
                .path = try allocator.dupe(u8, path),
                .file = file,
                .size = stat.size,
            });

            total_size += stat.size;
        }
    }

    var total_copied: usize = 0;

    const buffer_size = if (args.options.buffer > 0)
        @as(usize, 1) << args.options.buffer
    else
        std.math.max(std.mem.page_size, std.math.min(std.math.floorPowerOfTwo(usize, total_size / 1000), 1 << 21));

    const buffer = try allocator.alloc(u8, buffer_size);
    defer allocator.free(buffer);
    var fifo = std.fifo.LinearFifo(u8, .Slice).init(buffer);

    var progress = try Progress(@TypeOf(stderr)).init(total_size, stderr, .{
        .fps = args.options.fps,
        .window = args.options.window,
    });

    for (files.items) |info| {
        // LinearFifo.pump with status
        while (true) {
            if (fifo.writableLength() > 0) {
                const nr = try info.file.read(fifo.writableSlice(0));
                if (nr == 0) break;
                fifo.update(nr);
            }
            const nw = try stdout.write(fifo.readableSlice(0));
            total_copied += nw;
            fifo.discard(nw);

            try progress.update(total_copied);
        }
        while (fifo.readableLength() > 0) {
            const nw = try stderr.write(fifo.readableSlice(0));
            total_copied += nw;
            fifo.discard(nw);

            try progress.update(total_copied);
        }
    }

    return 0;
}

const FileInfo = struct {
    path: []const u8,
    file: std.fs.File,
    size: u64,
};

fn Progress(comptime Writer: type) type {
    return struct {
        const Self = @This();

        pub const Options = struct {
            fps: u64 = 10,
            window: u64 = 60,
        };

        total: usize,
        writer: Writer,

        frame_ns: u64,
        weight_factor: f64,

        timer: std.time.Timer,
        last_value: u64 = 0,
        last_ns: u64 = 0,
        ns_rate: f64 = 0,

        pub fn init(total: usize, writer: Writer, options: Options) !Self {
            var progress = Self{
                .total = total,
                .writer = writer,

                .frame_ns = std.time.ns_per_s / options.fps,
                .weight_factor = -1.0 / @intToFloat(f64, options.window * std.time.ns_per_s),

                .timer = try std.time.Timer.start(),
            };

            try progress.print(0, 0);
            return progress;
        }

        pub fn update(self: *Self, value: usize) !void {
            const now = self.timer.read();

            const last_frame_num = self.last_ns / self.frame_ns;
            const curr_frame_num = now / self.frame_ns;

            if (curr_frame_num > last_frame_num or value >= self.total) {
                self.updateRate(value, now);

                try self.print(value, now);

                self.last_value = value;
                self.last_ns = now;
            }
        }

        fn updateRate(self: *Self, value: usize, now: u64) void {
            const value_diff = @intToFloat(f64, value - self.last_value);
            const ns_diff = @intToFloat(f64, now - self.last_ns);
            const ns_rate = value_diff / ns_diff;
            const weight = 1.0 - std.math.exp(ns_diff * self.weight_factor);

            if (self.ns_rate == 0) {
                self.ns_rate = ns_rate;
            } else {
                self.ns_rate += weight * (ns_rate - self.ns_rate);
            }
        }

        fn print(self: *Self, value: usize, now: u64) !void {
            const writer = self.writer.writer();

            const progress = @intToFloat(f64, value) / @intToFloat(f64, self.total);

            try writer.print("\x1b[G{:.2} / {:.2} ({:.2}%)", .{
                fmtIntSizeBin(@truncate(u64, value)),
                fmtIntSizeBin(@truncate(u64, self.total)),
                fmtPrecision(100 * progress),
            });

            if (now > 0) {
                try writer.print(" in {:.2} ({:.2}/s)", .{
                    fmtDuration(now),
                    fmtIntSizeBin(@floatToInt(u64, self.ns_rate * std.time.ns_per_s)),
                });
            }

            if (value >= self.total) {
                try writer.writeAll("\x1b[K");
            } else if (self.ns_rate > 0) {
                const bytes_remaining = self.total - value;
                const ns_remaining = @floatToInt(u64, @intToFloat(f64, bytes_remaining) / self.ns_rate);

                try writer.print(", ETA {:.2}\x1b[K", .{fmtDuration(ns_remaining)});
            }

            try self.writer.flush();
        }
    };
}

fn formatDuration(ns: u64, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
    _ = fmt;

    const odd_units = [_]struct { ns: u64, suffix: u8 }{
        .{ .ns = 365 * std.time.ns_per_day, .suffix = 'y' },
        .{ .ns = 365 * std.time.ns_per_day / 12, .suffix = 'm' },
        .{ .ns = std.time.ns_per_week, .suffix = 'w' },
        .{ .ns = std.time.ns_per_day, .suffix = 'd' },
        .{ .ns = std.time.ns_per_hour, .suffix = 'h' },
        .{ .ns = std.time.ns_per_min, .suffix = 'm' },
        .{ .ns = std.time.ns_per_s, .suffix = 's' },
    };

    for (odd_units[0 .. odd_units.len - 1]) |unit, i| {
        if (ns < unit.ns) continue;

        // Print this unit
        try std.fmt.formatInt(ns / unit.ns, 10, .lower, options, writer);
        try writer.writeByte(unit.suffix);

        // Print the next unit
        const remaining = ns % unit.ns;
        const next = odd_units[i + 1];
        try std.fmt.formatInt(remaining / next.ns, 10, .lower, .{ .width = 2, .fill = '0' }, writer);
        try writer.writeByte(next.suffix);

        return;
    }

    const dec_units = [_]struct { ns: u64, suffix: []const u8 }{
        .{ .ns = std.time.ns_per_s, .suffix = "s" },
        .{ .ns = std.time.ns_per_ms, .suffix = "ms" },
        .{ .ns = std.time.ns_per_us, .suffix = "us" },
    };

    for (dec_units) |unit, i| {
        if (ns < unit.ns) continue;

        var float = @intToFloat(f64, ns) / @intToFloat(f64, unit.ns);
        var suffix = unit.suffix;
        if (float >= 999.5) {
            // Will be rounded up to 1000; promote unit
            float = 1;
            suffix = dec_units[i - 1].suffix;
        }

        try formatFloatPrecision(float, options, writer);
        try writer.writeAll(suffix);
        return;
    }

    try std.fmt.formatInt(ns, 10, .lower, .{}, writer);
    try writer.writeAll("ns");
}

test "formatDuration" {
    const cases = [_]struct { ns: u64, s: []const u8 }{
        .{ .ns = 1, .s = "1ns" },
        .{ .ns = 999, .s = "999ns" },
        .{ .ns = 1_000, .s = "1.00us" },
        .{ .ns = 1_004, .s = "1.00us" },
        .{ .ns = 1_005, .s = "1.01us" },
        .{ .ns = 1_010, .s = "1.01us" },
        .{ .ns = 99_040, .s = "99.0us" },
        .{ .ns = 99_050, .s = "99.1us" },
        .{ .ns = 998_499, .s = "998us" },
        .{ .ns = 998_500, .s = "999us" },
        .{ .ns = 999_499, .s = "999us" },
        .{ .ns = 999_500, .s = "1.00ms" },
        .{ .ns = 999_499_999, .s = "999ms" },
        .{ .ns = 999_500_000, .s = "1.00s" },
        .{ .ns = std.time.ns_per_s - 1, .s = "1.00s" },
        .{ .ns = std.time.ns_per_min - 1, .s = "60.0s" },
        .{ .ns = std.time.ns_per_min, .s = "1m00s" },
        .{ .ns = std.time.ns_per_min + 1, .s = "1m00s" },
        .{ .ns = std.time.ns_per_min + std.time.ns_per_s, .s = "1m01s" },
        .{ .ns = 2 * 365 * std.time.ns_per_day - 1, .s = "1y11m" },
        .{ .ns = 2 * 365 * std.time.ns_per_day, .s = "2y00m" },
        .{ .ns = 2 * 365 * std.time.ns_per_day + 1, .s = "2y00m" },
        .{ .ns = 2 * 365 * std.time.ns_per_day + 61 * std.time.ns_per_day, .s = "2y02m" },
    };

    for (cases) |case| {
        var buffer: [128]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        try formatDuration(case.ns, "", .{ .precision = 2 }, stream.writer());
        std.debug.print("{d} = {s}\n", .{ case.ns, buffer[0..stream.pos] });
        try std.testing.expectEqualStrings(case.s, buffer[0..stream.pos]);
    }
}

fn fmtDuration(ns: u64) std.fmt.Formatter(formatDuration) {
    return .{ .data = ns };
}

fn formatFloatPrecision(value: f64, options: std.fmt.FormatOptions, writer: anytype) !void {
    if (!std.math.isFinite(value)) return std.fmt.formatFloatDecimal(value, options, writer);
    const precision = @intToFloat(f64, options.precision orelse return std.fmt.formatFloatDecimal(value, options, writer));
    const exponent = std.math.floor(std.math.log10(value));

    var opts = options;
    opts.precision = if (exponent > precision)
        0
    else
        @floatToInt(usize, std.math.min(precision, precision - exponent));
    return std.fmt.formatFloatDecimal(value, opts, writer);
}

fn formatPrecision(value: f64, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
    _ = fmt;

    return formatFloatPrecision(value, options, writer);
}

fn fmtPrecision(value: f64) std.fmt.Formatter(formatPrecision) {
    return .{ .data = value };
}

test "formatFloatPrecision" {
    const cases = [_]struct { n: f64, s: []const u8 }{
        .{ .n = 0.0011111, .s = "0.00" },
        .{ .n = 0.011111, .s = "0.01" },
        .{ .n = 0.11111, .s = "0.11" },
        .{ .n = 1.11111, .s = "1.11" },
        .{ .n = 11.11111, .s = "11.1" },
        .{ .n = 111.11111, .s = "111" },
        .{ .n = 1111.11111, .s = "1111" },
        .{ .n = 11111.11111, .s = "11111" },
    };

    for (cases) |case| {
        var buffer: [128]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        try formatFloatPrecision(case.n, .{ .precision = 2 }, stream.writer());
        try std.testing.expectEqualStrings(case.s, buffer[0..stream.pos]);
    }
}

fn FormatSizeImpl(comptime radix: comptime_int) type {
    return struct {
        fn f(
            value: u64,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;

            if (value == 0) {
                return writer.writeAll("0B");
            }

            const mags_si = " kMGTPEZY";
            const mags_iec = " KMGTPEZY";

            const log2 = std.math.log2(value);
            const magnitude = switch (radix) {
                1000 => std.math.min(log2 / comptime std.math.log2(1000), mags_si.len - 1),
                1024 => std.math.min(log2 / 10, mags_iec.len - 1),
                else => unreachable,
            };
            var new_value = std.math.lossyCast(f64, value) / std.math.pow(f64, std.math.lossyCast(f64, radix), std.math.lossyCast(f64, magnitude));
            var suffix = switch (radix) {
                1000 => mags_si[magnitude],
                1024 => mags_iec[magnitude],
                else => unreachable,
            };
            if (new_value >= 999.5) {
                new_value /= radix;
                suffix = switch (radix) {
                    1000 => mags_si[magnitude + 1],
                    1024 => mags_iec[magnitude + 1],
                    else => unreachable,
                };
            }

            try formatFloatPrecision(new_value, options, writer);

            if (suffix == ' ') {
                return writer.writeAll("B");
            }

            const buf = switch (radix) {
                1000 => &[_]u8{ suffix, 'B' },
                1024 => &[_]u8{ suffix, 'i', 'B' },
                else => unreachable,
            };
            return writer.writeAll(buf);
        }
    };
}

const formatSizeDec = FormatSizeImpl(1000).f;
const formatSizeBin = FormatSizeImpl(1024).f;

/// Return a Formatter for a u64 value representing a file size.
/// This formatter represents the number as multiple of 1000 and uses the SI
/// measurement units (kB, MB, GB, ...).
pub fn fmtIntSizeDec(value: u64) std.fmt.Formatter(formatSizeDec) {
    return .{ .data = value };
}

/// Return a Formatter for a u64 value representing a file size.
/// This formatter represents the number as multiple of 1024 and uses the IEC
/// measurement units (KiB, MiB, GiB, ...).
pub fn fmtIntSizeBin(value: u64) std.fmt.Formatter(formatSizeBin) {
    return .{ .data = value };
}
