module BRAM_read_FSM_simple#(
    // parameters
    parameter DATA_WIDTH=10,
    parameter ADDR_WIDTH=10,
    parameter NUM_MODULE_INPUT=4,
    parameter START_ADDRESS=32'h8000_0000 
    )(
    // system signals
    input logic clk,                // system clock
    input logic reset,              // system reset
    
    // BRAM signals
    output logic [ADDR_WIDTH:0] BRAM_address,
    input logic  [DATA_WIDTH:0] BRAM_data,
    output logic enable,
    
    // control signals
    input logic read_start,
    output logic read_done,
    
    // module signals
    output logic [NUM_MODULE_INPUT-1:0][DATA_WIDTH-1:0] module_inputs);
    
    typedef enum {idle, read, done} state_types;
    state_types state, next_state;
    
    logic [NUM_MODULE_INPUT-1:0][DATA_WIDTH-1:0] next_module_inputs;
    logic [ADDR_WIDTH:0] next_BRAM_address;
    
    integer index, next_index;
    
    always_ff @(posedge clk, posedge reset)
        if (reset) begin
            state <= idle;
            BRAM_address <= START_ADDRESS;
            module_inputs <= 'd0;
            index <= 0;
        end
        else begin
            state <= next_state;
            BRAM_address <= next_BRAM_address;
            module_inputs <= next_module_inputs;
            index <= next_index;
        end
        
    always_comb begin
        next_state = idle;
        next_BRAM_address = BRAM_address;
        next_module_inputs = module_inputs;
        next_index = index;
        enable = 0;
        read_done = 0;
        unique case (state)
            idle: begin
                if (read_start) begin
                    next_state = read;
                    enable = 1;
                end
            end
            read: begin
                if (index >= NUM_MODULE_INPUT) 
                    next_state = done;
                else begin
                    next_module_inputs[index][DATA_WIDTH-1:0] = BRAM_data;
                    next_index = index + 1;
                    next_BRAM_address = BRAM_address + 1;
                    enable = 1;
                    next_state = read;
                end
            end
            done: begin
                read_done = 1;
            end
        endcase    
    end
endmodule
