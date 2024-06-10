module bram_sim(
    input logic clk,
    input logic reset,
    input logic[31:0] address,
    output logic[31:0] data_out,
    input logic enable
);

    logic [31:0] next_data_out;
    
    always_ff @(posedge clk)
        if (reset) 
            data_out = 32'hFFFFFFFF;
        else
            data_out = next_data_out;
    
    always_comb begin
        if (enable) begin
            unique case (address) 
                32'hB000_0000: next_data_out = 32'h11102220;
                32'hB000_0004: next_data_out = 32'h33304440;
                32'hB000_0008: next_data_out = 32'h55506660;
                32'hB000_000C: next_data_out = 32'h77708880;
                32'hB000_0010: next_data_out = 32'h9990AAA0;
                32'hB000_0014: next_data_out = 32'hBBB0CCC0;
                32'hB000_0018: next_data_out = 32'hBBB0CCC0;
                32'hB000_001C: next_data_out = 32'hDDD0EEE0;
                32'hB000_0020: next_data_out = 32'hFFF00000;
                32'hB000_0024: next_data_out = 32'h11223344;
                32'hB000_0028: next_data_out = 32'h55667788;
                32'hB000_002C: next_data_out = 32'h99AABBCC;
                32'hB000_0030: next_data_out = 32'hDDEEFF00;
                32'hB000_0034: next_data_out = 32'h01020304;
                32'hB000_0038: next_data_out = 32'h05060708;
                32'hB000_003C: next_data_out = 32'h090A0B0C;
                32'hB000_0040: next_data_out = 32'h0D0E0F00;
                32'hB000_0044: next_data_out = 32'h10203040;
                32'hB000_0048: next_data_out = 32'h50607080;
                32'hB000_004C: next_data_out = 32'h90A0B0C0;
                32'hB000_0050: next_data_out = 32'hD0E0F000;
                32'hB000_0054: next_data_out = 32'h1A2B3C4D;
                32'hB000_0058: next_data_out = 32'h2B3C4D5E;
                32'hB000_005C: next_data_out = 32'h3C4D5E6F;
                32'hB000_0060: next_data_out = 32'h4D5E6F70;
                32'hB000_0064: next_data_out = 32'h5E6F7081;
                32'hB000_0068: next_data_out = 32'h6F708192;
                32'hB000_006C: next_data_out = 32'h708192A3;
                32'hB000_0070: next_data_out = 32'h8192A3B4;
                32'hB000_0074: next_data_out = 32'h92A3B4C5;
                32'hB000_0078: next_data_out = 32'hA3B4C5D6;
                32'hB000_007C: next_data_out = 32'hB4C5D6E7;
                32'hB000_0080: next_data_out = 32'hC5D6E7F8;
                32'hB000_0084: next_data_out = 32'hD6E7F809;
                32'hB000_0088: next_data_out = 32'hE7F8091A;
                32'hB000_008C: next_data_out = 32'hF8091A2B;
                32'hB000_0090: next_data_out = 32'h091A2B3C;
                32'hB000_0094: next_data_out = 32'h1A2B3C4D;
                32'hB000_0098: next_data_out = 32'h2B3C4D5E;
                32'hB000_009C: next_data_out = 32'h3C4D5E6F;
                32'hB000_00A0: next_data_out = 32'h4D5E6F70;
                32'hB000_00A4: next_data_out = 32'h5E6F7081;
                32'hB000_00A8: next_data_out = 32'h6F708192;
                32'hB000_00AC: next_data_out = 32'h708192A3;
                32'hB000_00B0: next_data_out = 32'h8192A3B4;
                32'hB000_00B4: next_data_out = 32'h92A3B4C5;
                32'hB000_00B8: next_data_out = 32'hA3B4C5D6;
                32'hB000_00BC: next_data_out = 32'hB4C5D6E7;
                32'hB000_00C0: next_data_out = 32'hC5D6E7F8;
                32'hB000_00C4: next_data_out = 32'hD6E7F809;
                32'hB000_00C8: next_data_out = 32'hE7F8091A;
                32'hB000_00CC: next_data_out = 32'hF8091A2B;
                32'hB000_00D0: next_data_out = 32'h091A2B3C;
                32'hB000_00D4: next_data_out = 32'h1A2B3C4D;
                32'hB000_00D8: next_data_out = 32'h2B3C4D5E;
                32'hB000_00DC: next_data_out = 32'h3C4D5E6F;
                32'hB000_00E0: next_data_out = 32'h4D5E6F70;
                32'hB000_00E4: next_data_out = 32'h5E6F7081;
                32'hB000_00E8: next_data_out = 32'h6F708192;
                32'hB000_00EC: next_data_out = 32'h708192A3;
                32'hB000_00F0: next_data_out = 32'h8192A3B4;
                32'hB000_00F4: next_data_out = 32'h92A3B4C5;
                32'hB000_00F8: next_data_out = 32'hA3B4C5D6;
                32'hB000_00FC: next_data_out = 32'hB4C5D6E7;
                32'hB000_0100: next_data_out = 32'hC5D6E7F8;
                32'hB000_0104: next_data_out = 32'hD6E7F809;
                32'hB000_0108: next_data_out = 32'hE7F8091A;
                32'hB000_010C: next_data_out = 32'hF8091A2B;
            endcase
        end
        else next_data_out = 4;
    end
endmodule