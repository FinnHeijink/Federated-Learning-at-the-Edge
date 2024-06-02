module filter_fsm #(
  parameter width = 16,
  parameter input_width = 8,
  parameter im_dim = 28,
  parameter k_size = 9, // 3x3
  parameter k_dim = 3 // 3x3 kernel
  )(
  input  logic                  clk_i,
  input  logic                  rst_ni,
  
  input logic [input_width-1:0] pixel_i,
    
  input logic                   pix_data_valid,
  
  output logic           [7:0]  pixel_o,
  
  output logic                  conv_finished,
  
  output logic					kernel_constructed,
  
  input logic [143:0] k_val
);

  typedef enum {
      init, 
      commence_construction,
      construct_kernel, 
      await_buffers,
	  preprocess,
      process, 
	  stall_conv,
      done
  } state_t; 
    
  state_t state_d, state_q; 
  
  typedef enum {
    idle, 
    convolve,
	stall_reading,
    finished  
  } read_state;

  read_state r_state_d, r_state_q;
  
  // logic [2:0][2:0][input_width-1:0] pixel_i;
  logic [2:0][2:0][width-1:0] kernel_q, kernel_d, k_out; // 3x3 kernel in registers, 16-bit resolution
  logic [2:0][2:0][input_width-1:0] pixel_grid;
  logic commence, k_constructed, r_en;
  
  logic [$clog2(k_size)-1:0] k_address;
  
  logic [1:0] conv_stall; 
  
  logic conv_data_valid;
  logic [1:0] increment_buffer;
  
  logic [$clog2(im_dim)-1:0] row_pix_counter; // 5 bit signal
  logic [$clog2(im_dim*im_dim)-1:0] pix_counter; // 10 bit signal (28^2)
  
  logic [1:0] active_write_buffer;
  logic [1:0] active_read_buffer;
  
  logic p1_read, p2_read, p3_read, p4_read, p1_valid, p2_valid, p3_valid, p4_valid;
  logic read_buffer;
  logic [3:0] buff_valid;
  logic [3:0] buff_read;
  logic [4:0] read_counter;
  
  logic [2:0][input_width-1:0] p1_buffer, p2_buffer, p3_buffer, p4_buffer;
  
  logic [$clog2(im_dim)-1:0] read_row_counter; // Tracks total number of read rows committed to buffers, to know when an image is entirely read!
  
  assign kernel_constructed = k_constructed;
  
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      state_q <= init;  
      r_state_q <= idle; 
      kernel_q <= '0; 
    end else begin
      state_q <= state_d;
      r_state_q <= r_state_d;
      kernel_q <= kernel_d; 
    end 
  end
  
  // Counts the # of pixel in 'active row'
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      row_pix_counter <= '0;  
    end else if (pix_data_valid && row_pix_counter == 'd27) begin
      row_pix_counter <= '0;
    end else if (pix_data_valid) begin
      row_pix_counter <= row_pix_counter + 1;
	  if (conv_stall > 'd0) begin
	    conv_stall <= conv_stall - 1;
		increment_buffer <= increment_buffer + 1;
	  end 
    end 
  end
  
  always_ff @(posedge clk_i) begin 
    if (increment_buffer == 2'b10) begin
	  increment_buffer <= 'd0;
	end 
  end 

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      pix_counter <= '0;  
    end else if (pix_data_valid) begin
      pix_counter <= pix_counter + 1;
    end 
  end

  // Tracks the row buffer that corresponds with the given row being written to ('active write row')
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      active_write_buffer <= '0;  // start with p1
    end else if (row_pix_counter == 'd27 && pix_data_valid) begin
      active_write_buffer <= active_write_buffer + 1; // exploit overflow to loop back to 0!
    end
  end

  always_comb begin
    buff_valid = '0;
    buff_valid[active_write_buffer] = pix_data_valid; // Validity signal determines which row buffer stores input pixel
	{p4_valid, p3_valid, p2_valid, p1_valid} = buff_valid;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      read_counter <= '0;  
      read_row_counter <= '0;
	  conv_stall <= '0;
	  increment_buffer <= '0;
    end else if (read_buffer && read_counter == 'd25 && conv_stall == 'd0) begin
      read_counter <= '0;
      read_row_counter <= read_row_counter + 1; 
	  conv_stall <= 'd1;
	  increment_buffer <= 2'b01;
    end else if (read_buffer && conv_stall == 'd0) begin
      read_counter <= read_counter + 1; // counts every row during active convolution phase, similar to row_pix_counter
    end
  end
  

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      active_read_buffer <= '0;
    end else if (increment_buffer == 2'b10) begin
      active_read_buffer <= active_read_buffer + 1;  
    end
  end

  always_comb begin
    case(active_read_buffer)
      2'b00: begin
        pixel_grid = {p1_buffer, p2_buffer, p3_buffer};
      end
      2'b01: begin
        pixel_grid = {p2_buffer, p3_buffer, p4_buffer};
      end
      2'b10: begin
        pixel_grid = {p3_buffer, p4_buffer, p1_buffer};
      end
      2'b11: begin
        pixel_grid = {p4_buffer, p1_buffer, p2_buffer};
      end
      default: begin
        pixel_grid = '0;
      end
    endcase
  end

  always_comb begin
    case(r_state_q) // @suppress "Default clause missing from case statement"
      idle: begin
	    read_buffer <= 1'b0;
        if (pix_counter >= 83) begin // active_write_buffer ought to be at state 2'b11
          r_state_d <= convolve; // First 3 lines have been read 
        end else begin
          r_state_d <= idle;
        end
      end 
      
      convolve: begin
        read_buffer <= 1'b1; 
        if (read_row_counter >= 'd26) begin
          r_state_d <= finished;
        end else if (conv_stall > 'd0) begin 
		  r_state_d <= stall_reading;
		end else begin 
          r_state_d <= convolve;
        end
      end
	  
	  stall_reading: begin 
	    read_buffer <= 1'b0;
		if (conv_stall == 'd0) begin 
		  r_state_d <= convolve;
		end else begin 
		  r_state_d <= stall_reading;
		end 
	  end
      
      finished: begin
        read_buffer <= 1'b0;
        r_state_d <= finished;  
      end
    endcase
  end

  always_comb begin
    case(state_q)
      init: begin
        kernel_d <= kernel_q;
        commence <= 1'b0; 
        conv_data_valid <= 1'b0;
        state_d <= commence_construction; 
      end
      
      commence_construction: begin
        kernel_d <= kernel_q;
        commence <= 1'b1; 
        conv_data_valid <= 1'b0;
        state_d <= construct_kernel; 
      end
      
      construct_kernel: begin // Does BRAM-reader FSM provide kernel in correct form? [2:0][2:0][15:0]
        kernel_d <= k_out;     
        commence <= 1'b0;
        conv_data_valid <= 1'b0;
        if (k_constructed == 1'b1) begin
          state_d <= await_buffers;   
        end else begin
          state_d <= construct_kernel;
        end
      end
      
      await_buffers: begin
        kernel_d <= kernel_q;
        commence <= 1'b0;
        conv_data_valid <= 1'b0;
        if (read_buffer == 1'b1) begin
          state_d <= process;
        end else begin
          state_d <= await_buffers;
        end
      end
	  
	  preprocess: begin
	    kernel_d <= kernel_q; 
        commence <= 1'b0;    
        conv_data_valid <= 1'b0;
		state_d <= process;
	  end 
      
      process: begin
        kernel_d <= kernel_q; 
        commence <= 1'b0;    
        conv_data_valid <= 1'b1;
        if (r_state_q == finished && conv_finished == 1'b1) begin
          state_d <= done;
        end else if (conv_stall > 'd0) begin
		  state_d <= stall_conv;
		end else begin
          state_d <= process;
        end
      end
	  
	  stall_conv: begin
	    kernel_d <= kernel_q; 
		commence <= 1'b0;
		conv_data_valid <= 1'b0; 
		if (conv_stall == 'd0) begin
		  state_d <= preprocess;
		end else begin 
		  state_d <= stall_conv;
		end 
	  end
	  
      done: begin
        kernel_d <= kernel_q;
        commence <= 1'b0;
        conv_data_valid <= 1'b0;
        state_d <= done;
      end
    endcase
  end

  // pipelined filter!
  filter conv_layer (
    .clk_i(clk_i), 
    .rst_ni(rst_ni), 
    // .switch_i(switch_i),
    .pixel_i(pixel_grid),
    .data_valid(conv_data_valid),
    .k(kernel_q),
    .pixel_o(pixel_o),
    .output_valid(conv_finished)  
  );
  
/*
  // Kernel BRAM
  kernel_weight  #(
    .width(width),
    .k_size(k_size)
  ) k_bram (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .r_en(r_en),
    .address(k_address), 
    .k_out(k_val)  
  );
 */
  
  // Reads k_BRAM and returns kernel in proper form, takes 2 clk-cycles
  kernel_constructor #(
    .width(width),
    .k_size(k_size)
  ) k_const (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .commence(commence),
      .r_en(r_en),
      .k_val(k_val),
      .k_address(k_address),
      .kernel(k_out),
      .finished(k_constructed)
  );
  
  // The following "pixel buffers" function as row buffers, buffering row-data for the kernel
  
  pixel_buffer p1 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .pix_data(pixel_i),
    .data_valid(p1_valid),
	.read_address(read_counter),
    .pix_buffer(p1_buffer)  
  );
  
  pixel_buffer p2 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .pix_data(pixel_i),
    .data_valid(p2_valid),
	.read_address(read_counter),
    .pix_buffer(p2_buffer)  
  );
  
  pixel_buffer p3 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .pix_data(pixel_i),
    .data_valid(p3_valid),
	.read_address(read_counter),
    .pix_buffer(p3_buffer)
  );
  
  pixel_buffer p4 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .pix_data(pixel_i),
    .data_valid(p4_valid),
	.read_address(read_counter),
    .pix_buffer(p4_buffer)  
  );
    
endmodule : filter_fsm