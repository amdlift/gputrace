use gputrace_testkit::{Harness, target_bin};

#[test]
fn a_targets_calls_reach_the_stub_driver() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(
        run.stub_calls(),
        [
            "cuInit",
            "cuDriverGetVersion",
            "cuDeviceGetCount",
            "cuDeviceGetName"
        ]
    );
}

#[test]
fn the_stub_answers_deterministically() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stub_smoke"), &[])
        .expect("run target");

    assert_eq!(
        run.stdout(),
        "driver=13040\ndevices=1\nname=gputrace-stub-device-0\n"
    );
}

#[test]
fn the_stub_round_trips_a_copy_through_device_memory() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("alloc_copy"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(
        run.stdout(),
        "device_ptr=0x10000000\nreceived=deadbeef01020304\n"
    );
    assert_eq!(
        run.stub_calls(),
        [
            "cuInit",
            "cuDeviceGet",
            "cuCtxCreate_v4",
            "cuMemAlloc_v2",
            "cuMemcpyHtoD_v2",
            "cuMemcpyDtoH_v2",
            "cuMemFree_v2",
            "cuCtxDestroy_v2",
        ]
    );
}

#[test]
fn the_stub_binds_kernels_to_the_module_that_defines_them() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("module_launch"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(
        run.stdout(),
        concat!(
            "module=0x10100\n",
            "kernel_a=0x10200\n",
            "kernel_a_again=0x10200\n",
            "kernel_b=0x10300\n",
            "absent=500\n",
            "launch=0\n",
            "launch_after_unload=400\n",
        )
    );
    assert_eq!(
        run.stub_calls(),
        [
            "cuInit",
            "cuDeviceGet",
            "cuCtxCreate_v4",
            "cuModuleLoadData",
            "cuModuleGetFunction",
            "cuModuleGetFunction",
            "cuModuleGetFunction",
            "cuModuleGetFunction",
            "cuLaunchKernel",
            "cuModuleUnload",
            "cuLaunchKernel",
        ]
    );
}

#[test]
fn the_stub_orders_events_across_streams() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("stream_event"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(
        run.stdout(),
        concat!(
            "producer=0x10100\n",
            "consumer=0x10200\n",
            "start=0x10300\n",
            "finish=0x10400\n",
            "elapsed=1\n",
            "record_on_destroyed=400\n",
        )
    );
    assert_eq!(
        run.stub_calls(),
        [
            "cuInit",
            "cuDeviceGet",
            "cuCtxCreate_v4",
            "cuStreamCreate",
            "cuStreamCreate",
            "cuEventCreate",
            "cuEventCreate",
            "cuEventRecord",
            "cuStreamWaitEvent",
            "cuEventRecord",
            "cuStreamSynchronize",
            "cuStreamSynchronize",
            "cuEventSynchronize",
            "cuEventElapsedTime_v2",
            "cuEventDestroy_v2",
            "cuStreamDestroy_v2",
            "cuEventRecord",
        ]
    );
}

#[test]
fn the_stub_serves_the_dispatch_table() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("proc_address"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(
        run.stdout(),
        concat!(
            "lookup=0 status=0\n",
            "devices=1\n",
            "mem_alloc_is_v2=true\n",
            "per_thread_default_stream=true\n",
            "missing=500 status=1 null=true\n",
            "too_old=500 status=2\n",
            "v1=0 agrees=true\n",
        )
    );
    assert_eq!(
        run.stub_calls(),
        [
            "cuInit",
            "cuGetProcAddress_v2",
            "cuDeviceGetCount",
            "cuGetProcAddress_v2",
            "cuGetProcAddress_v2",
            "cuGetProcAddress_v2",
            "cuGetProcAddress_v2",
            "cuGetProcAddress",
        ]
    );
}

#[test]
fn every_host_thread_reaches_the_stub() {
    let harness = Harness::new().expect("harness");
    let run = harness
        .run(target_bin("multi_thread"), &[])
        .expect("run target");

    assert!(run.status().success(), "target failed: {}", run.stderr());
    assert_eq!(run.stdout(), "round_trips=4\n");

    assert_eq!(run.stub_call_count("cuInit"), 1);
    for symbol in [
        "cuCtxCreate_v4",
        "cuMemAlloc_v2",
        "cuMemcpyHtoD_v2",
        "cuMemcpyDtoH_v2",
        "cuMemFree_v2",
        "cuCtxDestroy_v2",
    ] {
        assert_eq!(run.stub_call_count(symbol), 4, "{symbol}");
    }
}
