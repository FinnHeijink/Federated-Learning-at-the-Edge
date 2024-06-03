module pixel_buffer #(
  parameter input_width = 8,
  parameter buff_size = 3,
  parameter im_dim = 28 // assuming 28 for im_dim
)(
  input logic clk_i,
  input logic rst_ni,  
  input logic [input_width-1:0] pix_data,
  input logic data_valid,
 // input logic data_read,
  input logic [$clog2(im_dim)-1:0] read_address,
  output logic [buff_size-1:0][input_width-1:0] pix_buffer
);

  logic [$clog2(im_dim)-1:0] write_address;
  logic [input_width-1:0] line_data [im_dim-1:0];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      write_address <= '0;
      // read_address <= '0;
      line_data <= '{default: '0};
    end else begin
      if (data_valid) begin
        line_data[write_address] <= pix_data;
        write_address <= (write_address == im_dim-1) ? '0 : write_address + 1;
      end /*
      if (data_read) begin
        read_address <= (read_address == 25) ? '0 : read_address + 1;
      end */
    end
  end

  always_comb begin
    pix_buffer = '{default: '0};
    for (int i = 0; i < buff_size; i++) begin
      pix_buffer[i] = line_data[(read_address + 2 - i)];
    end
  end

endmodule
