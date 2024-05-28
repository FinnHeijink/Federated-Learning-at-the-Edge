module kernel_constructor #(
  parameter width = 16,
  parameter k_size = 9
  )(
  input logic       clk_i,
  input logic       rst_ni,
  
  input logic       commence,
  output logic      r_en,
  
  input logic [143:0] k_val,
    
  output logic [$clog2(k_size)-1:0] k_address,
  output logic [2:0][2:0][width-1:0] kernel,
  
  output logic      finished
);

  typedef enum {
    init,
    init_loader,
    kernel_loader
  } state_t;
  
  state_t state_q, state_d; 
  
  logic count_up_addr;
  logic [$clog2(k_size)-1:0] addr_holder;
  


  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      state_q <= init; 
    end else begin
      state_q <= state_d;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
	  addr_holder <= '0;
	end
    if (count_up_addr) begin
      addr_holder <= addr_holder + 1;
    end else begin
      addr_holder <= addr_holder;
    end
  end
  
  // -----------------------
  // Single-cycle read-out implementation 
  // -----------------------
  always_comb begin
    case(state_q)
      init: begin
        r_en <= 1'b0;
        count_up_addr <= 1'b0; // Counter and address output redundant for single-word kernel
        k_address <= addr_holder; 
        finished <= 1'b0;
        if (commence == 1'b1) begin
          state_d <= init_loader;
        end else begin
          state_d <= init;
        end        
      end
      
      init_loader: begin
        r_en <= 1'b1; // By next clock cycle (state), k_bram will have read *all* weights 
        count_up_addr <= 1'b1;
        k_address <= addr_holder; 
        finished <= 1'b0;
        state_d <= kernel_loader;
      end
      
      kernel_loader: begin
        r_en <= 1'b0;
        count_up_addr <= 1'b0;
        k_address <= addr_holder; 
        finished <= 1'b1; 
        kernel <= k_val;
        state_d <= init;
      end 
      
      default: state_d <= state_q;
        
    endcase
  end // End single-cycle read-out implementation

  /* -- Multi-cycle case -- (Unfinished, hopefully unnecessary)
  always_comb begin
    case(state_q)
      init: begin
        r_en <= 1'b0;
        count_up_addr <= 1'b0;
        k_address <= addr_holder;
        finished <= 1'b0;
        if (commence == 1'b1) begin
          state_q <= init_loader;
        end else begin
          state_q <= init;
        end        
      end
    
      init_loader: begin
        r_en <= 1'b1; // By next clock cycle (state), k_bram will have read first weight
        count_up_addr <= 1'b0;
        k_address <= addr_holder; 
        finished <= 1'b0;
        state_q <= kernel_loader;
      end
  
      default: state_q <= state_d;
    endcase
  end
  --- Multi-cycle case -- */
 
endmodule : kernel_constructor