/*   MLIP is a software for Machine Learning Interatomic Potentials
 *   MLIP is released under the "New BSD License", see the LICENSE file.
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_UTILS_MLP_H
#define MLIP_UTILS_MLP_H

#include <string>

const std::string USAGE = \
"Usage:\n"
"mlp help                 prints this message\n"
"mlp list                 lists all the available commands\n"
"mlp help [command]       prints the decription of the command\n"
"mlp [command] [options]  executes the command (with the options)\n";

#define BEGIN_COMMAND(command_name, descr, usage) \
    if(command == "list") \
        std::cout << "    "  << command_name << ": " << descr << '\n'; \
    else if((args.size()==1) && command == "help" && (args[0] == command_name)) \
        {std::cout << "Usage: \n" << usage; is_command_found = true; } \
    else if (command == command_name) { is_command_found = true;

#define BEGIN_UNDOCUMENTED_COMMAND(command_name, descr, usage) \
    if(command == "list-undocumented") \
        std::cout << "    "  << command_name << ": " << descr << '\n'; \
    else if((args.size()==1) && command == "help" && (args[0] == command_name)) \
        std::cout << "UNDOCUMENTED COMMAND!!!\nUsage: \n" << usage; \
    else if (command == command_name) { is_command_found = true;

#define END_COMMAND }

#endif // MLP
